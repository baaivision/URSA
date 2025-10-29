# Copyright 2024 UC Berkeley Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
class DDPMSchedulerOutput(BaseOutput):
    """Output class for the scheduler's `step` function output."""

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999, alpha_transform_type="cosine"):
    """Create a beta schedule that discretizes the given alpha_t_bar function."""
    if alpha_transform_type == "cosine":
        alpha_bar_fn = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa
    elif alpha_transform_type == "exp":
        alpha_bar_fn = lambda t: math.exp(t * -12.0)  # noqa
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def rescale_zero_terminal_snr(betas):
    """Rescales betas to have zero terminal SNR."""
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas


class DDPMScheduler(SchedulerMixin, ConfigMixin):
    """
    `DDPMScheduler` explores the connections between denoising score matching and Langevin dynamics sampling.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            An array of betas to pass directly to the constructor without using `beta_start` and `beta_end`.
        variance_type (`str`, defaults to `"fixed_small"`):
            Clip the variance when adding noise to the denoised sample. Choose from `fixed_small`, `fixed_small_log`,
            `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """  # noqa

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        rescale_betas_zero_snr: int = False,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "scaled_linear":
            a, b = beta_start**0.5, beta_end**0.5
            self.betas = torch.linspace(a, b, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":  # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":  # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")
        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.init_noise_sigma = 1.0
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(num_train_timesteps)[::-1].copy())
        self.variance_type = variance_type

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
        """Scale the denoising model input depending on the current timestep."""
        return sample

    def sample_timesteps(self, size, device=None):
        return torch.randint(0, self.config.num_train_timesteps, size, device=device)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """Sets the discrete timesteps used for the diffusion chain (to be run before inference)."""
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")
        self.custom_timesteps = timesteps is not None
        self.num_inference_steps = num_inference_steps
        if timesteps is not None:
            timesteps = np.array(timesteps, dtype=np.int64)
        # See Table 2. of https://arxiv.org/abs/2305.08891
        elif self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
            timesteps = timesteps.round()[::-1].copy().astype(np.int64)
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = np.arange(0, num_inference_steps) * step_ratio
            timesteps = timesteps.round()[::-1].copy().astype(np.int64) + self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            timesteps = np.arange(self.config.num_train_timesteps, 0, -step_ratio)
            timesteps = timesteps.round().astype(np.int64) - 1
        else:
            raise ValueError(f"{self.config.timestep_spacing} is not supported.")
        self.timesteps = torch.as_tensor(timesteps, device=device)

    def _get_variance(self, t, predicted_variance=None):
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)  # noqa
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)
        if self.config.variance_type == "fixed_small_log":  # for rl-diffuser
            return torch.exp(0.5 * variance.log())
        elif self.config.variance_type == "fixed_large":
            return current_beta_t
        elif self.config.variance_type == "fixed_large_log":  # Glide max_log
            return torch.log(current_beta_t)
        elif self.config.variance_type == "learned":
            return predicted_variance
        elif self.config.variance_type == "learned_range":
            frac = (predicted_variance + 1) / 2
            min_log, max_log = variance.log(), torch.log(current_beta_t)
            return frac * max_log + (1 - frac) * min_log
        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """  # noqa
        t = timestep
        prev_t = self.previous_timestep(t)

        predicted_variance = None
        if self.variance_type in ("learned", "learned_range"):
            if model_output.shape[1] == sample.shape[1] * 2:
                model_output, predicted_variance = model_output.chunk(2, dim=1)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        elif self.config.prediction_type == "sample":
            pred_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_sample = alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
        else:
            raise ValueError(f"Unsupported prediction type given as {self.config.prediction_type}.")

        # 4. Compute coefficients for pred_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_sample_coeff = alpha_prod_t_prev**0.5 * current_beta_t / beta_prod_t
        current_sample_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        prev_sample = pred_sample_coeff * pred_sample + current_sample_coeff * sample

        # 6. Add noise
        if t > 0:
            device, dtype = model_output.device, model_output.dtype
            noise = randn_tensor(sample.shape, generator=generator, device=device, dtype=dtype)
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance)
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance).mul(0.5).exp()
            else:
                variance = self._get_variance(t, predicted_variance) ** 0.5
            prev_sample.add_(noise.mul_(variance))

        if not return_dict:
            return (prev_sample,)
        return DDPMSchedulerOutput(prev_sample=prev_sample)

    def previous_timestep(self, timestep):
        if self.custom_timesteps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                return torch.tensor(-1)
            return self.timesteps[index + 1]
        num_inference_steps = self.num_inference_steps or self.config.num_train_timesteps
        return timestep - self.config.num_train_timesteps // num_inference_steps

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        timesteps = timesteps.to(device=original_samples.device)
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        expand_shape = timesteps.shape + (1,) * (noise.dim() - timesteps.dim())
        sqrt_alpha_prod = sqrt_alpha_prod.view(expand_shape)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(expand_shape)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def get_velocity(
        self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        timesteps = timesteps.to(sample.device)
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        expand_shape = timesteps.shape + (1,) * (noise.dim() - timesteps.dim())
        sqrt_alpha_prod = sqrt_alpha_prod.view(expand_shape)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(expand_shape)
        return sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample

    def __len__(self):
        return self.config.num_train_timesteps
