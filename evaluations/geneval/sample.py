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
"""GenEval sampling for URSA models."""

import argparse
import collections
import json
import os
import os.path as osp

from tqdm import tqdm
import torch
import torch.distributed as dist

from diffnext.pipelines import URSAPipeline


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="geneval sampling")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint file")
    parser.add_argument("--prompt", type=str, default=None, help="prompt json file")
    parser.add_argument("--prompt_type", type=str, default="dense_prompt", help="prompt type")
    parser.add_argument("--height", type=int, default=1024, help="image height")
    parser.add_argument("--width", type=int, default=1024, help="image width")
    parser.add_argument("--guidance_scale", type=float, default=7, help="guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="inference steps")
    parser.add_argument("--prompt_size", type=int, default=4, help="prompt size for each batch")
    parser.add_argument("--sample_size", type=int, default=4, help="sample size for each prompt")
    parser.add_argument("--vae_batch_size", type=int, default=1, help="vae batch size")
    parser.add_argument("--distributed", action="store_true", help="distrbuted mode?")
    parser.add_argument("--outdir", type=str, default="", help="write to")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rank, world_size = 0, 1
    if args.distributed:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    device, dtype = torch.device("cuda", rank), torch.float16
    torch.cuda.set_device(device), torch.manual_seed(1337 + rank)
    generator = torch.Generator(device).manual_seed(1337 + rank)

    # Data.
    args.prompt = args.prompt if args.prompt else osp.join(osp.dirname(__file__), "prompts.json")
    meta = [json.loads(line) for line in open(osp.join(osp.dirname(__file__), "metadata.jsonl"))]
    prompt_list = json.load(open(args.prompt))[rank::world_size]

    # Arguments.
    gen_args = {"guidance_scale": args.guidance_scale}
    gen_args["vae_batch_size"] = args.vae_batch_size
    gen_args["num_inference_steps"] = args.num_inference_steps
    gen_args["height"], gen_args["width"] = args.height, args.width

    # Pipeline.
    pipe = URSAPipeline.from_pretrained(args.ckpt, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    for step in tqdm(range(0, len(prompt_list), args.prompt_size), disable=rank):
        samples, gen_args["generator"] = [], generator
        prompts = [_[args.prompt_type] for _ in prompt_list[step : step + args.prompt_size]]
        out_ids = [_["id"] for _ in prompt_list[step : step + args.prompt_size]] * args.sample_size
        [samples.extend(pipe(prompts, **gen_args).frames) for _ in range(args.sample_size)]
        name_cnt = collections.defaultdict(int)
        for i, img in enumerate(samples):
            prompt_root = f"{args.outdir}/{out_ids[i]:05d}"
            os.makedirs(f"{prompt_root}/samples", exist_ok=True)
            img.save(f"{prompt_root}/samples/{name_cnt[prompt_root]:05d}.png")
            name_cnt[prompt_root] += 1
            json.dump(meta[out_ids[i]], open(f"{prompt_root}/metadata.jsonl", "w"))
    (dist.barrier(device_ids=[rank]), dist.destroy_process_group()) if world_size > 1 else None
