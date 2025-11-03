# Copyright (c) 2024-present, BAAI. All Rights Reserved.
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
# ------------------------------------------------------------------------
"""Video generation pipeline for URSA."""

from typing_extensions import Self

import numpy as np
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch

from diffnext.image_processor import VaeImageProcessor
from diffnext.pipelines.pipeline_utils import URSAPipelineOutput, PipelineMixin


class URSAPipeline(DiffusionPipeline, PipelineMixin):
    """URSA video generation pipeline."""

    _optional_components = ["transformer", "scheduler", "vae", "tokenizer"]
    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer=None,
        scheduler=None,
        vae=None,
        tokenizer=None,
        trust_remote_code=True,
    ):
        super(URSAPipeline, self).__init__()
        self.vae = self.register_module(vae, "vae")
        self.tokenizer = self.register_module(tokenizer, "tokenizer")
        self.transformer = self.register_module(transformer, "transformer")
        self.scheduler = self.register_module(scheduler, "scheduler")
        self.vae_temporal_stride = self.vae.config.get("temporal_stride", 4)
        self.vae_spatial_stride = self.vae.config.get("spatial_stride", 8)
        self.tokenizer_args = {"padding": "max_length", "padding_side": "left", "truncation": True}
        self.image_processor = VaeImageProcessor()

    @property
    def _device(self) -> torch.device:
        """Return the execution device."""
        return getattr(self, "_offload_device", self.device)

    def to(self, *args, **kwargs) -> Self:
        for v in list(args) + list(kwargs.values()):
            self.scheduler.to(device=v) if isinstance(v, torch.device) else None
        return super().to(*args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        height=320,
        width=512,
        num_frames=1,
        num_inference_steps=25,
        guidance_scale=7,
        guidance_trunc=0.9,
        flow_shift=None,
        image=None,
        video=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        latents_shape=None,
        cond_latents=None,
        cond_indices=None,
        cond_noise_scale=0,
        prompt_ids=None,
        negative_prompt_ids=None,
        output_type="pil",
        max_prompt_length=320,
        vae_batch_size=1,
        **kwargs,
    ) -> URSAPipelineOutput:
        """The call function to the pipeline for generation.

        Args:
            prompt (str or List[str], *optional*):
                The prompt to be encoded.
            negative_prompt (str or List[str], *optional*):
                The prompt or prompts to guide what to not include in image generation.
            height (int, *optional*, defaults to 320)
                 The height in pixels of the generated video.
            width (int, *optional*, defaults to 512)
                 The width in pixels of the generated video.
            num_frames (int, *optional*, defaults to 49)
                The number of frames in the generated video.
            num_inference_steps (int, *optional*, defaults to 25):
                The number of inference steps.
            guidance_scale (float, *optional*, defaults to 7):
                The classifier guidance scale.
            guidance_trunc (float, *optional*, defaults to 0):
                The truncation threshold to classifier guidance.
            flow_shift (float, *optional*)
                The specified shift value for the timestep schedule.
            image (numpy.ndarray, *optional*):
                The image to be encoded as condition.
            video (numpy.ndarray, *optional*):
                The video to be encoded as condition.
            num_images_per_prompt (int, *optional*, defaults to 1):
                The number of images (or videos) that should be generated per prompt.
            generator (torch.Generator, *optional*):
                The random generator.
            latents (torch.Tensor, *optional*)
                The encoded latents.
            latents_shape (List[int], *optional*)
                The latents shape for (num_frames, height, width).
            cond_latents (torch.Tensor, *optional*)
                The encoded condition latents.
            cond_indices (List[int], *optional*)
                The valid indices to condition latents.
            cond_noise_scale (float, *optional*, defaults to 0)
                The specified level of noise added to condition latents.
            prompt_ids (torch.Tensor, *optional*)
                The precomputed prompt embeddings ids.
            negative_prompt_ids (torch.Tensor, *optional*)
                The precomputed negative prompt embedding ids.
            output_type (str, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            max_prompt_length (int, *optional*, defaults to 320)
                The maximum prompt length for truncation.
            vae_batch_size (int, *optional*, defaults to 1)
                The maximum batch size for decoding latents.

        Returns:
            URSAPipelineOutput: The pipeline output.
        """
        # 1. Check inputs
        if latents_shape is None:
            latents_shape = [(num_frames - 1) // self.vae_temporal_stride + 1]
            latents_shape += [height // self.vae_spatial_stride, width // self.vae_spatial_stride]
        latents_shape = ([1] if len(latents_shape) == 2 else []) + list(latents_shape)
        args = locals()

        # 2. Encode prompts
        txt_ids, neg_ids = self.encode_prompt(**dict(_ for _ in args.items() if "prompt" in _[0]))

        # 3. Prepare latent variables
        latents = self.prepare_latents(
            batch_size=len(txt_ids),
            latents_shape=latents_shape,
            image=image,
            video=video,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            cond_latents=cond_latents,
            cond_indices=cond_indices,
            cond_noise_scale=cond_noise_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_shift(flow_shift) if flow_shift else None
        self.scheduler.to(self._device).set_timesteps(num_inference_steps)

        # 5. Prepare RoPE positions
        txt_pos = torch.arange(max_prompt_length, device=self.device).view(-1, 1).expand(-1, 3)
        blk_pos = self.transformer.model.flex_rope.get_pos(latents_shape, txt_pos.size(0))

        # 6. Denoising loop
        bov_token_id = self.transformer.config.bov_token_id
        latent_shift = self.transformer.config.lm_vocab_size
        latents, num_latent_tokens = latents.flatten(1), latents.shape[1:].numel()
        img_ids = torch.nn.functional.pad(latents + latent_shift, (1, 0), value=bov_token_id)
        input_ids, uncond_ids = map(lambda _: torch.cat([_, img_ids], 1), (txt_ids, neg_ids))
        model_args = {"rope_pos": torch.cat([txt_pos, blk_pos[0]]).to(latents.device)}
        for timestep in self.progress_bar(self.scheduler.timesteps):
            t = self.scheduler.timestep_to_t(timestep)
            uncond_ids[:, -num_latent_tokens:] = input_ids[:, -num_latent_tokens:]
            if guidance_scale > 1:
                model_input = torch.cat([input_ids, uncond_ids])
                cond, uncond = self.transformer(model_input, **model_args)[0].chunk(2)
                z = uncond.add_(cond.sub_(uncond).mul_(guidance_scale if t < guidance_trunc else 1))
            else:
                z = self.transformer(input_ids, **model_args)[0]
            z = z[:, -(num_latent_tokens + 1) : -1]
            latents = self.scheduler.step(z, timestep, latents, generator=generator)[0]
            input_ids[:, -num_latent_tokens:] = latents + latent_shift

        # 7. Postprocessing
        video = latents = latents.view(-1, *latents_shape)
        output_type = "np" if output_type == "pil" and video.size(1) > 1 else output_type
        if output_type != "latent":
            video = self.image_processor.decode_latents(self.vae, latents, vae_batch_size)
        video = self.image_processor.postprocess(video, output_type)
        return URSAPipelineOutput(frames=video)

    def prepare_latents(
        self,
        batch_size=None,
        latents_shape=None,
        image=None,
        video=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        cond_latents=None,
        cond_indices=None,
        cond_noise_scale=0,
    ) -> torch.Tensor:
        """Prepare the latents for generation.

        Args:
            batch_size (int, *optional*)
                The batch size to latents.
            latents_shape (List[int], *optional*)
                The shape to latents.
            image (numpy.ndarray, *optional*):
                The image to be encoded.
            video (numpy.ndarray, *optional*):
                The video to be encoded.
            num_images_per_prompt (int, *optional*, defaults to 1):
                The number of images that should be generated per prompt.
            generator (torch.Generator, *optional*):
                The random generator.
            latents (torch.Tensor, *optional*)
                The encoded latents.
            cond_latents (torch.Tensor, *optional*)
                The encoded condition latents.
            cond_indices (List[int], *optional*)
                The available indices of condition latents.
            cond_noise_scale (float, *optional*, defaults to 0)
                The specified level of noise added to condition latents.
        Returns:
            torch.Tensor: The latents tensor.
        """
        if latents is None:
            latents = torch.ones(batch_size, *latents_shape[-3:], dtype=torch.long).to(self._device)
            latents.random_(to=self.scheduler.codebook_size, generator=generator)
        if image is not None:
            cond_latents = self.encode_image(image, num_images_per_prompt, generator)
        if video is not None:
            cond_latents = self.encode_video(video, num_images_per_prompt, generator)
        if cond_latents is not None and cond_noise_scale > 0:
            cond_latents = self.scheduler.add_noise(cond_latents, 1 - cond_noise_scale, generator)
        if cond_latents is not None and cond_indices is None:
            latents[:, : cond_latents.shape[1]] = cond_latents
        for i in cond_indices if (cond_indices and cond_latents is not None) else []:
            latents[:, i] = cond_latents[:, i]
        return latents

    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt=1,
        negative_prompt=None,
        prompt_ids=None,
        negative_prompt_ids=None,
        max_prompt_length=320,
    ) -> torch.Tensor:
        """Encode text prompts.

        Args:
            prompt (str or List[str], *optional*):
                The prompt to be encoded.
            num_images_per_prompt (int, *optional*, defaults to 1):
                The number of images that should be generated per prompt.
            negative_prompt (str or List[str], *optional*):
                The prompt or prompts not to guide the image generation.
            prompt_ids (torch.Tensor, *optional*)
                The precomputed prompt embeddings ids.
            negative_prompt_ids (torch.Tensor, *optional*)
                The precomputed negative prompt embedding ids.
            max_prompt_length (int, *optional*)
                The maximum prompt length for truncation.
        Returns:
            torch.Tensor: The prompt embedding ids.
        """

        def select_or_pad(a, b, n=1):
            return [a or b] * n if isinstance(a or b, str) else (a or b)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        negative_prompt = select_or_pad(negative_prompt, "", len(prompt))
        args = {"max_length": max_prompt_length, "return_tensors": "pt", **self.tokenizer_args}
        if prompt_ids is None:
            prompt_ids = self.tokenizer(prompt, **args).input_ids.to(self._device)
            prompt_ids = prompt_ids.repeat_interleave(num_images_per_prompt, 0)
        if negative_prompt_ids is None:
            negative_prompt_ids = self.tokenizer(negative_prompt, **args).input_ids.to(self._device)
            negative_prompt_ids = negative_prompt_ids.repeat_interleave(num_images_per_prompt, 0)
        return prompt_ids, negative_prompt_ids

    def encode_image(self, image, num_images_per_prompt=1, generator=None) -> torch.Tensor:
        """Encode image prompt.

        Args:
            image (numpy.ndarray):
                The image to be encoded, shape (bsz, h, w, 3) or (h, w, 3).
            num_images_per_prompt (int):
                The number of images that should be generated per prompt.
            generator (torch.Generator, *optional*):
                The random generator.

        Returns:
            torch.Tensor: The image embedding ids.
        """
        x = np.array(image) if not isinstance(image, np.ndarray) else image
        x = torch.as_tensor(x, device=self._device).to(self.dtype).sub(127.5).div_(127.5)
        x = x.unflatten(0, (-1, 1)).permute(0, 4, 1, 2, 3) if x.dim() == 4 else x
        x = x.unflatten(0, (1, 1, -1)).permute(0, 4, 1, 2, 3) if x.dim() == 3 else x
        x = self.vae.scale_(self.vae.encode(x).latent_dist.sample(generator))
        return x.expand(num_images_per_prompt, -1, -1, -1)

    def encode_video(self, video, num_videos_per_prompt=1, generator=None) -> torch.Tensor:
        """Encode video prompt.

        Args:
            video (numpy.ndarray):
                The video to be encoded, shape (bsz, t, h, w, 3) or (t, h, w, 3).
            num_images_per_prompt (int):
                The number of videos that should be generated per prompt.
            generator (torch.Generator, *optional*):
                The random generator.

        Returns:
            torch.Tensor: The video embedding ids.
        """
        x = torch.as_tensor(video, device=self._device).to(self.dtype).sub(127.5).div_(127.5)
        x = x.permute(0, 4, 1, 2, 3) if x.dim() == 5 else x
        x = x.unflatten(0, (1, -1)).permute(0, 4, 1, 2, 3) if x.dim() == 4 else x
        x = self.vae.scale_(self.vae.encode(x).latent_dist.sample(generator))
        return x.expand(num_videos_per_prompt, -1, -1, -1)
