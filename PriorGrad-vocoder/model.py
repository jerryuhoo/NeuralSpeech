# Copyright 2022 (c) Microsoft Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

# for wavegrad
from base import BaseModule
from downsampling import (
    DownsamplingBlock as DBlock,
)

from layers import (
    Conv1dWithInitialization,
)

from linear_modulation import (
    FeatureWiseLinearModulation as FiLM,
)

from upsampling import (
    UpsamplingBlock as UBlock,
)

import numpy as np


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(max_steps), persistent=False
        )
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, n_cond_global=None):
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        if n_cond_global is not None:
            self.conditioner_projection_global = Conv1d(
                n_cond_global, 2 * residual_channels, 1
            )
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step, conditioner_global=None):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        if conditioner_global is not None:
            y = y + self.conditioner_projection_global(conditioner_global)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class PriorGrad(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.use_prior = params.use_prior
        self.condition_prior = params.condition_prior
        self.condition_prior_global = params.condition_prior_global
        assert not (
            self.condition_prior and self.condition_prior_global
        ), "use only one option for conditioning on the prior"
        print("use_prior: {}".format(self.use_prior))
        self.n_mels = params.n_mels
        self.n_cond = None
        print("condition_prior: {}".format(self.condition_prior))
        if self.condition_prior:
            self.n_mels = self.n_mels + 1
            print("self.n_mels increased to {}".format(self.n_mels))
        print("condition_prior_global: {}".format(self.condition_prior_global))
        if self.condition_prior_global:
            self.n_cond = 1

        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        self.spectrogram_upsampler = SpectrogramUpsampler(self.n_mels)
        if self.condition_prior_global:
            self.global_condition_upsampler = SpectrogramUpsampler(self.n_cond)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    self.n_mels,
                    params.residual_channels,
                    2 ** (i % params.dilation_cycle_length),
                    n_cond_global=self.n_cond,
                )
                for i in range(params.residual_layers)
            ]
        )
        self.skip_projection = Conv1d(
            params.residual_channels, params.residual_channels, 1
        )
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        self.wavegrad = WaveGradNN(params)

        print(
            "num param: {}".format(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            )
        )

    def forward(
        self, audio, spectrogram, diffusion_step, noise_level, global_cond=None
    ):
        # x = self.input_projection(x)
        # x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        # spectrogram = self.spectrogram_upsampler(spectrogram)
        if global_cond is not None:
            global_cond = self.global_condition_upsampler(global_cond)

        skip = []

        # print("audio.shape: {}".format(audio.shape))
        # print("spectrogram.shape: {}".format(spectrogram.shape))
        # print("diffusion_step.shape: {}".format(diffusion_step.shape))
        # print("noise_level.shape: {}".format(noise_level.shape))
        x = self.wavegrad(mels=spectrogram, yn=audio, noise_level=noise_level)

        # for layer in self.residual_layers:
        #     x, skip_connection = layer(x, spectrogram, diffusion_step, global_cond)
        #     skip.append(skip_connection)

        # print("x.shape: {}".format(x.shape))

        # x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        # x = self.skip_projection(x)
        # x = F.relu(x)
        # x = self.output_projection(x)
        return x


class WaveGradNN(BaseModule):
    """
    WaveGrad is a fully-convolutional mel-spectrogram conditional
    vocoder model for waveform generation introduced in
    "WaveGrad: Estimating Gradients for Waveform Generation" paper (link: https://arxiv.org/pdf/2009.00713.pdf).
    The concept is built on the prior work on score matching and diffusion probabilistic models.
    Current implementation follows described architecture in the paper.
    """

    def __init__(self, config) -> None:
        super(WaveGradNN, self).__init__()
        # Building upsampling branch (mels -> signal)
        self.ublock_preconv = Conv1dWithInitialization(
            in_channels=80,
            out_channels=config.upsampling_preconv_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        upsampling_in_sizes = [
            config.upsampling_preconv_out_channels
        ] + config.upsampling_out_channels[:-1]
        self.ublocks = torch.nn.ModuleList(
            [
                UBlock(
                    in_channels=in_size,
                    out_channels=out_size,
                    factor=factor,
                    dilations=dilations,
                )
                for in_size, out_size, factor, dilations in zip(
                    upsampling_in_sizes,
                    config.upsampling_out_channels,
                    config.factors,
                    config.upsampling_dilations,
                )
            ]
        )
        self.ublock_postconv = Conv1dWithInitialization(
            in_channels=config.upsampling_out_channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Building downsampling branch (starting from signal)
        self.dblock_preconv = Conv1dWithInitialization(
            in_channels=1,
            out_channels=config.downsampling_preconv_out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        downsampling_in_sizes = [
            config.downsampling_preconv_out_channels
        ] + config.downsampling_out_channels[:-1]
        self.dblocks = torch.nn.ModuleList(
            [
                DBlock(
                    in_channels=in_size,
                    out_channels=out_size,
                    factor=factor,
                    dilations=dilations,
                )
                for in_size, out_size, factor, dilations in zip(
                    downsampling_in_sizes,
                    config.downsampling_out_channels,
                    config.factors[1:][::-1],
                    config.downsampling_dilations,
                )
            ]
        )
        # Building FiLM connections (in order of downscaling stream)
        film_in_sizes = [32] + list(config.downsampling_out_channels)
        film_out_sizes = list(config.upsampling_out_channels[::-1])
        film_factors = [1] + list(config.factors[1:][::-1])
        self.films = torch.nn.ModuleList(
            [
                FiLM(
                    in_channels=in_size,
                    out_channels=out_size,
                    input_dscaled_by=np.product(
                        film_factors[: i + 1]
                    ),  # for proper positional encodings initialization
                )
                for i, (in_size, out_size) in enumerate(
                    zip(film_in_sizes, film_out_sizes)
                )
            ]
        )

    def forward(self, mels, yn, noise_level):
        """
        Computes forward pass of neural network.
        :param mels (torch.Tensor): mel-spectrogram acoustic features of shape [B, n_mels, T//hop_length]
        :param yn (torch.Tensor): noised signal `y_n` of shape [B, T]
        :param noise_level (float): level of noise added by diffusion
        :return (torch.Tensor): epsilon noise
        """
        # Prepare inputs
        assert len(mels.shape) == 3  # B, n_mels, T
        yn = yn.unsqueeze(1)
        assert len(yn.shape) == 3  # B, 1, T

        # Downsampling stream + Linear Modulation statistics calculation
        statistics = []
        dblock_outputs = self.dblock_preconv(yn)
        scale, shift = self.films[0](x=dblock_outputs, noise_level=noise_level)
        statistics.append([scale, shift])
        for dblock, film in zip(self.dblocks, self.films[1:]):
            dblock_outputs = dblock(dblock_outputs)
            scale, shift = film(x=dblock_outputs, noise_level=noise_level)
            statistics.append([scale, shift])
        statistics = statistics[::-1]

        # Upsampling stream
        ublock_outputs = self.ublock_preconv(mels)
        for i, ublock in enumerate(self.ublocks):
            scale, shift = statistics[i]
            ublock_outputs = ublock(x=ublock_outputs, scale=scale, shift=shift)
        outputs = self.ublock_postconv(ublock_outputs)
        return outputs.squeeze(1)
