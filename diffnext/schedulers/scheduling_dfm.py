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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, esither express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Simple implementation of discrete flow matching schedulers."""

import dataclasses
import os
from typing import Union
from typing_extensions import Self

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin
import torch


@dataclasses.dataclass
class KineticOptimalSchedulerOutput(BaseOutput):
    """Output for scheduler's `step` function output."""

    prev_sample: torch.LongTensor


class DiscreteProbPath(object):
    """Define a general discrete probability path."""

    def __init__(self, emb):
        """Create a ``DiscreteProbPath``.

        Args:
            emb (Union[torch.Tensor, torch.nn.Embedding])
                The codebook embeddings.
        """
        self.generator = None
        self.emb = emb.weight if isinstance(emb, torch.nn.Embedding) else emb

    def categorical(self, prob) -> torch.Tensor:
        """Categorical sampling according to weights in the last dimension.

        Args:
            prob (torch.Tensor)
                The sample token probability, shape (bsz, ..., codebook_size).

        Returns:
            torch.Tensor: The sample token index, shape (bsz, ...).
        """
        return prob.flatten(0, -2).multinomial(1, generator=self.generator).view(*prob.shape[:-1])


class MixtureDiscreteProbPath(DiscreteProbPath):
    """Define a mixture discrete probability path."""

    def sample(self, x_1, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Sample from the affine probability path.

        Args:
            x_1 (torch.Tensor)
                The target token index, shape (bsz, ...).
            t (float or torch.Tensor)
                The timestep ``t``, shape (bsz,).

        Returns:
            torch.Tensor: The sample token index at time t, shape (bsz, ...).
        """
        t = t.to(self.emb).view([-1] + [1] * (x_1.dim() - 1)) if hasattr(t, "cpu") else t
        x_0 = x_1.new_empty(x_1.shape).random_(to=self.emb.shape[0], generator=self.generator)
        return x_0.where(t.new_empty(x_1.shape).uniform_(generator=self.generator).lt(1 - t), x_1)

    def get_velocity(self, logits, x_t, t: float, x_1=None) -> torch.Tensor:
        """Return the velocity by converting the factorized posterior.

        Args:
            logits (torch.Tensor)
                The sample token logits at time t+1, shape (bsz, ..., codebook_size).
            x_t (torch.Tensor)
                The sample token index at time t, shape (bsz, ...).
            t (float)
                The timestep ``t``.
            x_1 (torch.Tensor, optional):
                The sample token index at time t+1, shape (bsz, ...).

        Returns:
            torch.Tensor: The velocity ``v``.
        """
        x_1 = self.categorical(logits.softmax(-1)) if x_1 is None else x_1
        return logits.zero_().scatter_(-1, x_1.unsqueeze(-1), 1 / (1 - t))


class MetricDiscreteProbPath(DiscreteProbPath):
    """Define a metric-induced discrete probability path."""

    def __init__(self, emb, alpha=0.9, c=3, eps=1e-5):
        """Create a ``MetricDiscreteProbPath``.

        Args:
            emb (Union[torch.Tensor, torch.nn.Embedding])
                The codebook embeddings.
            alpha (float)
                The value to ``alpha``.
            c (float)
                The value to ``c``.
            eps (float, *optional*, defaults to 1e-5):
                A small value to clip the L2 normalization denominator.
        """
        self.alpha, self.c, self.eps, self.generator = alpha, c, eps, None
        emb = emb.weight if isinstance(emb, torch.nn.Embedding) else emb
        self.emb = torch.nn.functional.normalize(emb, dim=-1, eps=eps)
        self.emb_sumsq = self.emb.square().sum(-1)
        self.emb_mul2t = self.emb.mul(2).T.contiguous()

    def get_dist(self, emb_1: torch.Tensor, emb_2: torch.Tensor = None) -> torch.Tensor:
        """Return the distance between two input embeddings.

        Args:
            emb_1 (torch.Tensor)
                The input1 embeddings, shape (bsz, ..., dim).
            emb_2 (torch.Tensor, optional)
                The input2 embeddings, shape (bsz, ..., dim) or (bsz, ..., codebook_size).

        Returns:
            torch.Tensor: The distance, shape (bsz, ..., 1) or (bsz, ..., codebook_size).
        """
        emb_1 = torch.nn.functional.normalize(emb_1, dim=-1, eps=self.eps)
        if emb_2 is None or emb_1.size() != emb_2.size():  # Distance between input and codebook.
            emb_1_sumsq, emb_2_sumsq = emb_1.square().sum(-1, True), self.emb_sumsq
            return torch.add(emb_1_sumsq, emb_2_sumsq, out=emb_2).sub_(emb_1 @ self.emb_mul2t)
        emb_2 = torch.nn.functional.normalize(emb_2, dim=-1, eps=self.eps)
        return emb_1.sub(emb_2).abs_().square_().sum(-1, keepdim=True)

    def get_prob(self, emb: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Return the metric-induced probability.

        Args:
            emb (torch.Tensor)
                The input embeddings, shape (bsz, ..., dim).
            t (float or torch.Tensor)
                The timestep ``t``, shape (bsz,).

        Returns:
            torch.Tensor: The probability at timestep ``t``, shape (bsz, ..., codebook_size).
        """
        beta = self.c * (t / (1 - t)) ** self.alpha
        beta = beta.to(emb).view([-1] + [1] * (emb.dim() - 1)) if hasattr(t, "cpu") else beta
        return self.get_dist(emb).mul_(-beta).softmax(-1)

    def get_prob_by_dist(self, dist: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Return the metric-induced probability by distance.

        Args:
            dist (torch.Tensor)
                The distance, shape (bsz, ..., codebook_size).
            t (float or torch.Tensor)
                The timestep ``t``, shape (bsz,).

        Returns:
            torch.Tensor: The probability at timestep ``t``, shape (bsz, ..., codebook_size).
        """
        beta = self.c * (t / (1 - t)) ** self.alpha
        beta = beta.to(dist).view([-1] + [1] * (dist.dim() - 1)) if hasattr(t, "cpu") else beta
        return dist.mul(-beta).softmax(-1)

    def sample(self, x_1, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Sample from the affine probability path.

        Args:
            x_1 (torch.Tensor)
                The target token index, shape (bsz, ...).
            t (float or torch.Tensor)
                The timestep ``t``, shape (bsz,).

        Returns:
            torch.Tensor: The sample token index at time t, shape (bsz, ...).
        """
        return self.categorical(self.get_prob(self.emb[x_1], t))

    def get_velocity(self, logits, x_t, t: float, x_1=None) -> torch.Tensor:
        """Return the velocity by converting the factorized posterior.

        Args:
            logits (torch.Tensor)
                The sample token logits at time t+1, shape (bsz, ..., codebook_size).
            x_t (torch.Tensor)
                The sample token index at time t, shape (bsz, ...).
            t (float)
                The timestep ``t``.
            x_1 (torch.Tensor, optional):
                The sample token index at time t+1, shape (bsz, ...).

        Returns:
            torch.Tensor: The velocity ``v``, shape (bsz, ..., codebook_size).
        """
        numerator = self.c * self.alpha * (t ** (self.alpha - 1)) if t > 0 else 0
        d_beta_t = numerator / (1 - t) ** (self.alpha + 1)
        emb_x_1 = self.emb[self.categorical(logits.softmax(-1)) if x_1 is None else x_1]
        dist_x_1_x = self.get_dist(emb_x_1, logits)  # (bsz, ..., codebook_size)
        prob_x_1_x = self.get_prob_by_dist(dist_x_1_x, t)  # (bsz, ..., codebook_size)
        dist_x_t_x_1 = self.get_dist(self.emb[x_t], emb_x_1)  # (bsz, ..., 1)
        dist = torch.nn.functional.relu(dist_x_1_x.sub_(dist_x_t_x_1).neg_(), inplace=True)
        return prob_x_1_x.mul_(d_beta_t).mul_(dist)  # (bsz, ..., codebook_size)


class KineticOptimalScheduler(SchedulerMixin, ConfigMixin):
    """Kinetic optimal scheduler with general discrete paths."""

    @register_to_config
    def __init__(self, alpha=None, c=None, shift=1.0, eps=1e-5, **kwargs):
        self.alpha, self.c, self.shift, self.eps = alpha, c, shift, eps
        self.init_args, self.path, self.codebook_size = kwargs or {}, None, 0
        self.init_args.setdefault("shift", shift) if shift != 1 else None

    def __repr__(self) -> str:
        """Return the extra representation of this scheduler."""
        s = f"{self.__class__.__name__}"
        if self.alpha is None:  # Fallback to ``MixtureDiscreteProbPath``.
            return s + "(shift={shift})".format(**self.__dict__)
        return s + "(alpha={alpha}, c={c}, shift={shift})".format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, pretrained_path, device=None, dtype=None, **kwargs) -> Self:
        """Instantiate the scheduler from a pretrained model vocabulary."""
        return KineticOptimalScheduler().load_pretrained(pretrained_path, device, dtype, **kwargs)

    def load_pretrained(self, pretrained_path=None, device=None, dtype=None, **kwargs) -> Self:
        """Load the scheduler from a pretrained model vocabulary."""
        pretrained_path = self.init_args.get("pretrained_path", None) or pretrained_path
        pretrained_args = super().from_pretrained(pretrained_path, **kwargs).__dict__
        pretrained_args.update({"init_args": self.init_args, **self.init_args})
        self.__dict__.update(pretrained_args)
        model_file = os.path.join(pretrained_path, "scheduler_model.pth")
        emb = torch.load(model_file, weights_only=False)["path.emb"]
        emb = emb.to(device).to(dtype=dtype or torch.float16)
        self.path = MetricDiscreteProbPath(emb=emb, alpha=self.alpha, c=self.c, eps=self.eps)
        self.path = MixtureDiscreteProbPath(emb=emb) if self.alpha is None else self.path
        self.codebook_size = self.path.emb.size(0)
        return self

    def to(self, device=None, dtype=None) -> Self:
        """Convert to given device and dtype."""
        for k, v in self.path.__dict__.items():
            self.path.__dict__[k] = v.to(device, dtype) if isinstance(v, torch.Tensor) else v
        return self

    def sample_timesteps(self, size, device=None, generator=None) -> torch.Tensor:
        """Sample a batch of timesteps for training.

        Args:
            size (Tuple[int])
                The sample size of timesteps.
            device (torch.device, optional)
                The output device.
            generator (torch.Generator, optional):
                The random generator.
        """
        sigma = 1 - torch.rand(size, device=device, generator=generator).mul_(0.999)
        return 1 - self.shift * sigma / (1 + (self.shift - 1) * sigma)

    def set_timesteps(self, num_inference_steps, *args, **kwargs):
        """Set the inference timesteps for sampling.

        Args:
            num_inference_steps (int)
                The number of inference steps.
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps).tolist()

    def add_noise(self, original_samples, timesteps, generator=None) -> torch.Tensor:
        """Add forward noise to samples.

        Args:
            original_samples (torch.Tensor)
                The sample token index, shape (bsz, ...).
            t (float or torch.Tensor)
                The timestep ``t``, shape (bsz,).
            generator (torch.Generator, optional):
                The random generator.

        Returns:
            torch.Tensor: The sample token index at time t, shape (bsz, ...).
        """
        self.path.generator = generator if generator else self.path.generator
        return self.path.sample(original_samples, timesteps)

    def timestep_to_t(self, timestep) -> float:
        """Return the ``t`` for given timestep.

        Args:
            timestep (int)
                The discrete timestep index.

        Returns:
            float: The continuous timestep in [0, 1).
        """
        sigma = 1 - self.timesteps[timestep] / self.num_inference_steps
        return 1 - self.shift * sigma / (1 + (self.shift - 1) * sigma)

    def sample(self, model_output, generator=None) -> torch.Tensor:
        """Sample token index from the model logits.

        Args:
            model_output (torch.Tensor)
                The sample token logits, shape (bsz, ..., codebook_size).
            generator (torch.Generator, optional):
                The random generator.

        Returns:
            torch.Tensor: The sample token index, shape (bsz, ...).
        """
        self.path.generator = generator if generator else self.path.generator
        return self.path.categorical(model_output.softmax(-1))

    def step(
        self,
        model_output,
        timestep,
        sample,
        prev_sample=None,
        generator=None,
        return_dict=True,
    ) -> KineticOptimalSchedulerOutput:
        """Predict the sample from the previous timestep.

        Args:
            model_output (torch.Tensor)
                The sample token logits at time t+1, shape (bsz, ..., codebook_size).
            timestep (int)
                The discrete timestep index.
            sample (torch.Tensor)
                The sample token index at time t, shape (bsz, ...).
            prev_sample (torch.Tensor, optional)
                The sample token index at time t+1, shape (bsz, ...).
            generator (torch.Generator, optional):
                The random generator.
            return_dict (bool, optional)
                Whether return the output in a dict.

        Returns:
            torch.Tensor: The sample token index at time t+1, shape (bsz, ...).
        """
        self.path.generator = generator if generator else self.path.generator
        if timestep == self.num_inference_steps - 1:
            prev_sample = self.sample(model_output) if prev_sample is None else prev_sample
        else:
            t = self.timestep_to_t(timestep)
            dt = self.timestep_to_t(timestep + 1) - t
            v = self.path.get_velocity(model_output, sample, t, prev_sample)
            u_dist = torch.empty_like(sample, dtype=v.dtype).uniform_(generator=generator)
            jump_thresh = 1 - v.scatter_(-1, sample[..., None], 0).sum(-1).mul_(-dt).exp_()
            prev_sample, jump_index = sample.clone(), u_dist < jump_thresh
            prev_sample[jump_index] = self.path.categorical(v[jump_index])
        if not return_dict:
            return (prev_sample,)
        return KineticOptimalSchedulerOutput(prev_sample=prev_sample)
