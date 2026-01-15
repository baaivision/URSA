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
"""Generic training pipeline for URSA."""

import os
from typing import Dict
from typing_extensions import Self

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import numpy as np
import torch
from torch.nn.functional import pad as pad_func

from diffnext.pipelines.pipeline_utils import PipelineMixin


class URSATrainPipeline(DiffusionPipeline, PipelineMixin):
    """Pipeline for training URSA models."""

    _optional_components = ["transformer", "scheduler", "vae", "tokenizer"]

    def __init__(
        self,
        transformer=None,
        scheduler=None,
        vae=None,
        tokenizer=None,
        trust_remote_code=True,
    ):
        super(URSATrainPipeline, self).__init__()
        self.train_config, self.accelerator, self.logger = None, None, None
        self.vae = self.register_module(vae, "vae")
        self.tokenizer = self.register_module(tokenizer, "tokenizer")
        self.transformer = self.register_module(transformer, "transformer")
        self.scheduler = self.register_module(scheduler, "scheduler")

    @property
    def model(self) -> torch.nn.Module:
        """Return the trainable model."""
        return self.transformer

    def to(self, *args, **kwargs) -> Self:
        for v in list(args) + list(kwargs.values()):
            self.scheduler.to(device=v) if isinstance(v, torch.device) else None
        return super().to(*args, **kwargs)

    def configure_model(self, config, accelerator=None, logger=None) -> torch.nn.Module:
        """Configure the trainable model."""
        self.train_config, self.accelerator, self.logger = config, accelerator, logger
        ckpt, _ = config.model.get("gradient_checkpointing", 0), self.model.train()
        for layer in self.model.model.layers:
            setattr(layer, "gradient_checkpointing", ckpt >= 3)  # -> O3
            setattr(layer.self_attn, "gradient_checkpointing", 1 < ckpt < 3)  # -> O2
            setattr(layer.mlp, "gradient_checkpointing", 0 < ckpt < 3)  # -> O1
        self.model.pipeline_preprocess = self.preprocess  # Preprocess hook.
        self.model.pipeline_postprocess = self.postprocess  # Postprocess hook.
        if "lora" in self.train_config.model:  # Add PEFT.
            from peft import LoraConfig, PeftModel, get_peft_model

            lora_config = LoraConfig(**config.model.lora.params)
            lora_config.target_modules = list(lora_config.target_modules)  # Fix JSON serialization.
            if config.experiment.resume_iter > 0:
                resume_args = {"config": lora_config, "is_trainable": True}
                ckpt = os.path.join(config.experiment.resume_from_checkpoint, config.model.name)
                self.transformer = PeftModel.from_pretrained(self.model, ckpt, **resume_args)
            else:
                self.transformer = get_peft_model(self.model, lora_config)
        batch_size_per_gpu = config.training.batch_size
        seq_parallel_size = config.training.get("sequence_parallel_size", 1)
        batch_size = batch_size_per_gpu * accelerator.gradient_accumulation_steps
        batch_size *= accelerator.num_processes // seq_parallel_size
        logger.info(">>> " + str(self.scheduler))
        logger.info(f"Num training steps = {self.train_config.training.max_train_steps}")
        logger.info(f"Batch size = {batch_size_per_gpu} ({seq_parallel_size} devices)")
        logger.info(f"Gradient batch size = {batch_size}")
        logger.info(f"Gradient accumulation steps = {config.training.gradient_accumulation_steps}")
        return self.model

    def process_prompts(self, inputs: Dict):
        """Process text prompts."""
        prompts = inputs["prompt"]
        for i, (s, text) in enumerate(zip(inputs.get("motion", []), prompts)):
            prompts[i] = (f"motion={s:.1f}, " if np.random.rand() > 0.4 else "") + text
        prompts = ["" if np.random.rand() < 0.1 else x for x in prompts]
        tokenizer_args = {**self.train_config.model.tokenizer.params, "return_tensors": "pt"}
        inputs["txt_ids"] = self.tokenizer(prompts, **tokenizer_args).input_ids.to(self.device)

    def process_latents(self, inputs: Dict):
        """Process video latents."""
        x = torch.as_tensor(inputs.pop("latents"), device=self.device)
        x = x.to(dtype=self.dtype if x.is_floating_point() else torch.int64)
        inputs["img_ids"] = self.vae.scale_(self.vae.latent_dist(x).sample())

    def process_inputs(self, inputs):
        """Process model inputs."""
        bov_id, num_blocks = self.model.config.bov_token_id, 1
        inp_ids, img_ids = inputs["img_ids"], inputs["img_ids"]
        txt_ids, txt_len = inputs["txt_ids"], inputs["txt_ids"].size(1)
        thw, block_size = inp_ids.shape[1:], inp_ids.size(1) // num_blocks
        # Prepare block pos.
        txt_pos = torch.arange(txt_len, device=inp_ids.device).view(-1, 1).repeat(1, 3)
        blk_pos = self.model.model.flex_rope.get_pos((num_blocks, block_size) + thw[1:], txt_len)
        rope_pos = torch.cat([txt_pos, blk_pos.flatten(0, 1)])  # Packed.
        # Prepare block ids.
        if self.train_config.model.get("async_timestep", False):
            inp_ids = img_ids.flatten(0, 1)  # (B, T, H, W) -> (B * T, H, W)
        t = self.scheduler.sample_timesteps(inp_ids.shape[:1], device=img_ids.device)
        inp_ids = self.scheduler.add_noise(inp_ids, t).add(len(self.tokenizer)).view(img_ids.shape)
        img_ids = pad_func(img_ids.unflatten(1, (-1, block_size)).flatten(2), (1, 0), value=-100)
        inp_ids = pad_func(inp_ids.unflatten(1, (-1, block_size)).flatten(2), (1, 0), value=bov_id)
        inputs["input_ids"] = torch.cat([txt_ids, inp_ids.flatten(1)], 1)
        inputs["labels"] = torch.cat([txt_ids.new_full(txt_ids.shape, -100), img_ids.flatten(1)], 1)
        inputs["rope_pos"] = rope_pos.unsqueeze(0).expand(inp_ids.size(0), -1, -1).contiguous()
        block_lens = [txt_len + inp_ids.shape[2]] + [inp_ids.shape[2]] * (num_blocks - 1)
        self.model.flex_attn.set_offsets_by_lens(block_lens) if len(block_lens) > 1 else None

    def preprocess(self, inputs: Dict) -> Dict:
        """Define the pipeline preprocess at every call."""
        self.process_prompts(inputs)
        self.process_latents(inputs)
        self.process_inputs(inputs)

    def postprocess(self, loss: torch.Tensor, acc1: torch.Tensor) -> Dict:
        """Define the pipeline postprocess at every call."""
        outputs = {"loss": loss}
        num_metrics = self.train_config.training.get("num_metrics", self.accelerator.num_processes)
        outputs["metric/loss"] = self.accelerator.gather(loss.data)[:num_metrics]
        outputs["metric/acc1"] = self.accelerator.gather(acc1)[:num_metrics]
        return outputs
