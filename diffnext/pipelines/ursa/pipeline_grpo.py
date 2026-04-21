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
"""GRPO training pipeline for URSA."""

import contextlib
import gc
import os
from types import MethodType
from typing import Dict

from accelerate.utils import broadcast_object_list, gather_object
import numpy as np
import torch
from torch.nn.functional import pad as pad_func

from diffnext.engine.model_ema import ModelEMA
from diffnext.pipelines.ursa.pipeline_ursa import URSAPipeline
from diffnext.pipelines.ursa.pipeline_train import URSATrainPipeline
from diffnext.schedulers.scheduling_dfm import KineticOptimalSchedulerOutput
from diffnext.utils.omegaconf_utils import config_to_object


class GRPOState(object):
    """GRPO state."""

    def __init__(self):
        self.input_ids, self.output_ids, self.logps = None, None, None

    @staticmethod
    def approx_kl(logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
        """Return the approximate KL divergence used in DeepSeekMath (Sec 4.1.1)."""
        return torch.exp(ref_logps - logps) - (ref_logps - logps) - 1

    @staticmethod
    def entropy(logits: torch.Tensor) -> torch.Tensor:
        """Return entropy from logits."""
        logps = torch.nn.functional.log_softmax(logits, dim=-1)
        return -logps.exp().mul_(logps).sum(dim=-1).float()

    @staticmethod
    @torch.compile(dynamic=True)
    def get_logps(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Return the log probabilities along the index."""
        return torch.gather(logits.log_softmax(-1), -1, index.unsqueeze(-1)).squeeze(-1).float()

    @staticmethod
    def step(scheduler, model_output, timestep, sample, generator=None):
        """Diffusion step function with states."""
        scheduler.path.generator = generator if generator else scheduler.path.generator
        prev_sample = scheduler.path.categorical(model_output.softmax(-1))
        prev_sample.states = states = sample.__dict__.pop("states", [GRPOState()])
        states[-1].reset(sample, prev_sample)
        if timestep < scheduler.num_inference_steps - 1:
            t = scheduler.timestep_to_t(timestep)
            dt = scheduler.timestep_to_t(timestep + 1) - t
            v = scheduler.path.get_velocity(model_output, sample, t, prev_sample)
            u_dist = torch.empty_like(sample, dtype=v.dtype).uniform_(generator=generator)
            jump_thresh = 1 - v.scatter_(-1, sample[..., None], 0).sum(-1).mul_(-dt).exp_()
            prev_sample, jump_index = sample.clone(), u_dist < jump_thresh
            prev_sample[jump_index] = scheduler.path.categorical(v[jump_index])
            prev_sample.states = states.__iadd__([GRPOState()])
        return KineticOptimalSchedulerOutput(prev_sample)

    def reset(self, input_ids=None, output_ids=None, logits=None):
        """Set ``input_ids``, ``output_ids`` and ``logps``."""
        self.input_ids = self.input_ids if input_ids is None else input_ids
        self.output_ids = self.output_ids if output_ids is None else output_ids
        self.logps = self.logps if logits is None else self.get_logps(logits, self.output_ids)


class URSAGRPOPipeline(URSATrainPipeline, URSAPipeline):
    """Pipeline for training URSA models."""

    def __init__(
        self,
        transformer=None,
        scheduler=None,
        vae=None,
        tokenizer=None,
        trust_remote_code=True,
    ):
        URSAPipeline.__init__(self, transformer, scheduler, vae, tokenizer)
        self.ref_model, self.reward_fn = None, None
        self.train_config, self.accelerator, self.logger = None, None, None
        self.set_progress_bar_config(disable=True)  # Disable progress bar.

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
        if "lora" in config.model:  # Add PEFT.
            from peft import LoraConfig, PeftModel, get_peft_model

            lora_config = LoraConfig(**config.model.lora.params)
            lora_config.target_modules = list(lora_config.target_modules)  # Fix JSON serialization.
            if config.experiment.resume_iter > 0:
                resume_args = {"config": lora_config, "is_trainable": True}
                ckpt = os.path.join(config.experiment.resume_from_checkpoint, config.model.name)
                self.transformer = PeftModel.from_pretrained(self.model, ckpt, **resume_args)
            else:
                self.transformer = get_peft_model(self.model, lora_config)
            self.ref_model = self.transformer
        else:
            self.ref_model = ModelEMA(self.model, decay=1) if config.reward.kl_beta > 0 else None
        num_batches1, num_batches2 = config.training.num_batches, config.sampling.num_batches
        prompt_size = config.train_dataloader.params.batch_size
        prompt_size = prompt_size * config.sampling.num_images_per_prompt
        batch_size = prompt_size * config.sampling.train_steps
        batch_size = batch_size // (num_batches2 // num_batches1)
        sp_size = config.training.get("sequence_parallel_size", 1)
        dp_size = accelerator.num_processes // sp_size
        batch_size_per_gpu = prompt_size // dp_size // config.sampling.num_batches
        train_steps = config.sampling.train_steps
        if batch_size_per_gpu != config.training.batch_size:
            raise ValueError(f"Excepted config.training.batch_size = {batch_size_per_gpu}")
        logger.info(">>> " + str(self.scheduler))
        logger.info(f"Num training steps = {config.training.max_train_steps}")
        logger.info(f"Batch size = {batch_size_per_gpu} ({sp_size} devices)")
        logger.info(f"Reward batch size = {prompt_size} ({num_batches2} batches)")
        logger.info(f"Gradient batch size = {batch_size} ({num_batches1} * {train_steps} batches)")
        logger.info(f"Gradient accumulation steps = {config.training.gradient_accumulation_steps}")
        self.reward_fn = config_to_object(config.reward).to(self.device)
        self.scheduler.step = MethodType(GRPOState.step, self.scheduler)  # Patch ``step(...)``
        return self.model

    def process_prompts(self, inputs: Dict):
        """Process text prompts."""
        for i, (s, text) in enumerate(zip(inputs.get("motion", []), inputs["prompt"])):
            inputs["prompt"][i] = (f"motion={s:.1f}, " if np.random.rand() > 0.4 else "") + text
        prompts = inputs["prompt"] * self.train_config.sampling.num_images_per_prompt
        metadatas = inputs["metadata"] * self.train_config.sampling.num_images_per_prompt
        rank, world_size = self.accelerator.process_index, self.accelerator.num_processes
        prompts, metadatas = broadcast_object_list(prompts), broadcast_object_list(metadatas)
        inputs["prompt"] = list(np.array(prompts).reshape((world_size, -1))[rank])
        inputs["metadata"] = list(np.array(metadatas).reshape((world_size, -1))[rank])
        tokenizer_args = {**self.train_config.model.tokenizer.params, "return_tensors": "pt"}
        txt_ids = self.tokenizer(inputs["prompt"], **tokenizer_args).input_ids.to(self.device)
        inputs["txt_ids"] = txt_ids.unflatten(0, (-1, self.train_config.training.batch_size))

    def process_latents(self, inputs: Dict):
        """Process video latents."""
        for txt_ids in inputs["txt_ids"]:
            frames, states = self(prompt_ids=txt_ids, **self.train_config.sampling.rollout)[0]
            start = np.random.randint(*self.train_config.sampling.get("train_start", [0, 1]))
            steps, _ = self.train_config.sampling.train_steps, inputs["frames"].append(frames)
            inputs["old_logps"] += [_.logps for _ in states[start : start + steps]]
            if self.train_config.sampling.get("forward_process", True):
                for timestep in self.scheduler.timesteps[start : start + steps]:
                    t = self.scheduler.timestep_to_t(timestep)
                    inputs["states"] += [self.scheduler.add_noise(states[-1].output_ids, t)]
            else:
                inputs["states"] += [_.input_ids for _ in states[start : start + steps]]
            if self.train_config.sampling.get("clean_response", True):
                inputs["responses"] += [states[-1].output_ids] * steps
            else:
                inputs["responses"] += [_.output_ids for _ in states[start : start + steps]]
        frame_size = [_ // self.vae_spatial_stride for _ in inputs["frames"][-1].shape[-2:]]
        inputs["states"] = torch.stack(inputs["states"]).unflatten(-1, [-1] + frame_size)

    def process_rewards(self, inputs: Dict):
        """Process video rewards."""
        frames = torch.cat(inputs.pop("frames")).mul_(0.5).add_(0.5).clamp_(0, 1)  # => [0, 1]
        scores, values = self.reward_fn(frames, inputs["prompt"], inputs["metadata"])[:2]
        inputs.update({f"metric/reward_{k}": np.array(gather_object(v)) for k, v in values.items()})
        scores = np.array(gather_object(scores))
        scores = scores.reshape((self.train_config.sampling.num_images_per_prompt, -1))
        inputs["metric/scores"], inputs["metric/scores_std"] = scores, scores.std(axis=0)
        group_mean = scores.mean(axis=0, keepdims=True)
        std_axis = (0, 1) if self.train_config.reward.get("global_std", False) else 0
        inputs["advantages"] = (scores - group_mean) / (scores.std(std_axis, keepdims=True) + 1e-4)

    def process_batches(self, inputs: Dict):
        """Process video batches."""
        num_batches = inputs["txt_ids"].shape[0]
        num_timesteps = self.train_config.sampling.train_steps
        advantage_clipping = self.train_config.reward.advantage_clipping
        rank, world_size = self.accelerator.process_index, self.accelerator.num_processes
        advantages = inputs["advantages"].reshape((world_size, -1))[rank].reshape((num_batches, -1))
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = advantages.repeat_interleave(num_timesteps, 0)
        advantages = advantages.clamp(-advantage_clipping, advantage_clipping)
        inputs["txt_ids"] = inputs["txt_ids"].repeat_interleave(num_timesteps, 0)
        inputs["step"], inputs["labels"] = 0, {"advantages": advantages}

    def process_inputs(self, inputs: Dict):
        """Process model inputs."""
        bov_id, num_blocks = self.model.config.bov_token_id, 1
        state, step = inputs["states"][inputs["step"]], inputs["step"]
        txt_ids, txt_len = inputs["txt_ids"][step], inputs["txt_ids"][step].size(1)
        thw, block_size = state.shape[1:], state.size(1) // num_blocks
        txt_pos = torch.arange(txt_len, device=state.device).view(-1, 1).repeat(1, 3)
        blk_pos = self.model.model.flex_rope.get_pos((num_blocks, block_size) + thw[1:], txt_len)
        rope_pos = torch.cat([txt_pos, blk_pos.flatten(0, 1)])  # Packed.
        state = state + len(self.tokenizer)  # Shift.
        state = pad_func(state.unflatten(1, (-1, block_size)).flatten(2), (1, 0), value=bov_id)
        inputs["input_ids"] = inputs["ref_input_ids"] = torch.cat([txt_ids, state.flatten(1)], 1)
        inputs["rope_pos"] = rope_pos.unsqueeze(0).expand(state.size(0), -1, -1).contiguous()

    @torch.no_grad()
    def process_refers(self, inputs: Dict):
        """Process model references."""
        response = inputs["responses"][inputs["step"]]
        with getattr(self.ref_model, "disable_adapter", contextlib.nullcontext)():
            logits = self.ref_model(inputs["ref_input_ids"], rope_pos=inputs["rope_pos"]).sample
        inputs["ref_logps"] = GRPOState.get_logps(logits[:, -response.size(-1) - 1 : -1], response)

    def preprocess(self, inputs: Dict) -> Dict:
        """Define the pipeline preprocess at every call."""
        if "step" not in inputs:  # =========> Exploration
            self.process_prompts(inputs)  # => Map
            self.process_latents(inputs)  # => Rollout
            self.process_rewards(inputs)  # => Reduce
            self.process_batches(inputs)  # => Batching
            gc.collect()  # =================> Clean
        self.process_inputs(inputs)  # ======> Exploitation
        self.process_refers(inputs) if self.ref_model else None

    def postprocess(self, inputs: Dict, logits: torch.Tensor) -> Dict:
        """Define the pipeline postprocess at every call."""
        response = inputs["responses"][inputs["step"]]
        old_logps = inputs["old_logps"][inputs["step"]]
        advantages = inputs["labels"]["advantages"][inputs["step"]]
        ratio_clipping = self.train_config.reward.ratio_clipping
        logps = GRPOState.get_logps(logits[:, -response.size(-1) - 1 : -1], response).mean(1)
        ratio = torch.exp(logps - logps.data if old_logps is None else old_logps.mean(1))
        clip_surr = advantages * ratio.clamp(1 - ratio_clipping, 1 + ratio_clipping)
        clip_frac = ratio.data.sub(1).abs().gt(ratio_clipping).float()
        outputs = dict((k, v) for k, v in inputs.items() if k.startswith("metric/"))
        outputs["loss"] = policy_loss = -torch.minimum(advantages * ratio, clip_surr).mean()
        outputs["metric/policy_loss"] = self.accelerator.gather(policy_loss.data)
        outputs["metric/clip_frac"] = self.accelerator.gather(clip_frac.mean())
        outputs["metric/entropy"] = self.accelerator.gather(GRPOState.entropy(logits.data).mean())
        if self.ref_model:
            kl_loss = GRPOState.approx_kl(logps, inputs["ref_logps"].mean(1)).mean()
            outputs["loss"] += kl_loss * self.train_config.reward.kl_beta
            outputs["metric/kl_loss"] = self.accelerator.gather(kl_loss.data)
        inputs["step"] = (inputs["step"] + 1) % len(inputs["responses"])  # Step to the next batch.
        return outputs
