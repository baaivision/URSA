# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Image processor."""

from typing import List, Union

import numpy as np
import PIL.Image
import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin


class VaeImageProcessor(ConfigMixin):
    """Image processor for VAE."""

    def postprocess(
        self, image: torch.Tensor, output_type: str = "pil"
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        """Postprocess the image output from tensor.

        Args:
            image (torch.Tensor):
                The image tensor.
            output_type (str, *optional*, defaults to `pil`):
                The output image type, can be one of `pil`, `np`, `pt`, `latent`.

        Returns:
            Union[PIL.Image.Image, np.ndarray, torch.Tensor]: The postprocessed image.
        """
        if output_type == "latent" or output_type == "pt":
            return image
        image = self.pt_to_numpy(image)
        if output_type == "np":
            return image
        if output_type == "pil":
            return self.numpy_to_pil(image)
        return image

    @staticmethod
    @torch.no_grad()
    def decode_latents(vae: nn.Module, latents: torch.Tensor, vae_batch_size=1) -> torch.Tensor:
        """Decode VAE latents.

        Args:
            vae (torch.nn.Module):
                The VAE model.
            latents (torch.Tensor):
                The input latents.
            vae_batch_size (int, *optional*, defaults to 1)
                The maximum images in a batch to decode.

        Returns:
            torch.Tensor: The output tensor.

        """
        x, batch_size = vae.unscale_(latents), latents.size(0)
        sizes, splits = [vae_batch_size] * (batch_size // vae_batch_size), []
        sizes += [batch_size - sum(sizes)] if sum(sizes) != batch_size else []
        for x_split in x.split(sizes) if len(sizes) > 1 else [x]:
            splits.append(vae.decode(x_split).sample)
        return torch.cat(splits) if len(splits) > 1 else splits[0]

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """Convert images from a torch tensor to a numpy array.

        Args:
            images (torch.Tensor):
                The image tensor.

        Returns:
            np.ndarry: The image array.
        """
        x = images.permute(0, 2, 3, 4, 1) if images.dim() == 5 else images.permute(0, 2, 3, 1)
        return x.mul(127.5).add_(127.5).clamp(0, 255).byte().cpu().numpy()

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
        """Convert images from a numpy array to a list of PIL objects.

        Args:
            images (np.ndarray):
                The image array.

        Returns:
            List[PIL.Image.Image]: A list of PIL images.
        """
        images = images[None, ...] if images.ndim == 3 else images
        images = images.reshape((-1,) + images.shape[2:]) if images.ndim == 5 else images
        return [PIL.Image.fromarray(image) for image in images]
