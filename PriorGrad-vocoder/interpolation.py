# pyre-ignore-all-errors
# pylint: skip-file

import torch

from base import BaseModule  # @oss-only

# @fb-only: from langtech.tts.vocoders.models.src.wavegrad.base import BaseModule


class InterpolationBlock(BaseModule):
    def __init__(
        self,
        scale_factor,
        mode: str = "linear",
        align_corners: bool = False,
        downsample: bool = False,
    ) -> None:
        super(InterpolationBlock, self).__init__()
        self.downsample = downsample
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        outputs = torch.nn.functional.interpolate(
            x,
            size=x.shape[-1] * self.scale_factor
            if not self.downsample
            else x.shape[-1] // self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=False,
        )
        return outputs
