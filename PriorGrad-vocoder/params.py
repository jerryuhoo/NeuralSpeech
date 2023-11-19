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

import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


params = AttrDict(
    # Training params
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,
    use_l2loss=True,
    n_iterations=100000000,  # added from config
    grad_clip_threshold=1,  # added from config
    scheduler_step_size=7000,  # added from config
    scheduler_gamma=0.9,  # added from config
    noise_schedule_interval=10000,  # added from config
    # Data params
    sample_rate=22050,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    fmin=0,
    fmax=8000,
    crop_mel_frames=62,  # PriorGrad keeps the previous open-source implementation
    # new data params for PriorGrad-vocoder
    use_prior=True,
    # optional parameters to additionally use the frame-level energy as the conditional input
    # one can choose one of the two options as below. note that only one can be set to True.
    condition_prior=False,  # whether to use energy prior as concatenated feature with mel. default is false
    condition_prior_global=False,  # whether to use energy prior as global condition with projection. default is false
    # minimum std that clips the prior std below std_min. ensures numerically stable training.
    std_min=0.1,
    # whether to clip max energy to certain value. Affects normalization of the energy.
    # Lower value -> more data points assign to ~1 variance. so pushes latent space to higher variance regime
    # if None, no override, uses computed stat
    # for volume-normalized waveform with HiFi-GAN STFT, max energy of 4 gives reasonable range that clips outliers
    max_energy_override=4.0,
    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],  # T>=50
    # Added model settings from config
    # factors=[5, 5, 3, 2, 2], # modified this to make its product equal to 256
    factors=[4, 4, 4, 2, 2],
    upsampling_preconv_out_channels=768,
    upsampling_out_channels=[512, 512, 256, 128, 128],
    upsampling_dilations=[
        [1, 2, 1, 2],
        [1, 2, 1, 2],
        [1, 2, 4, 8],
        [1, 2, 4, 8],
        [1, 2, 4, 8],
    ],
    downsampling_preconv_out_channels=32,
    downsampling_out_channels=[128, 128, 256, 512],
    downsampling_dilations=[[1, 2, 4], [1, 2, 4], [1, 2, 4], [1, 2, 4]],
    training_noise_schedule={  # format adapted for params structure
        "n_iter": 1000,
        "betas_range": [1.0e-6, 0.01],
    },
    test_noise_schedule={  # format adapted for params structure
        "n_iter": 50,
        "betas_range": [1.0e-6, 0.01],
    },
)
