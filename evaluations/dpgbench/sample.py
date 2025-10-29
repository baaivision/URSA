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
"""DPGBench sampling for URSA models."""

import argparse
import collections
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from tqdm import tqdm
import torch
import torch.distributed as dist

from diffnext.pipelines import URSAPipeline


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="dpgbench sampling")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint file")
    parser.add_argument("--prompt", type=str, default=None, help="prompt json file")
    parser.add_argument("--prompt_type", type=str, default="prompt", help="prompt type")
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
    prompt_list = json.load(open(args.prompt))[rank::world_size]
    os.makedirs(args.outdir, exist_ok=True)

    # Arguments.
    gen_args = {"guidance_scale": args.guidance_scale, "output_type": "np"}
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
        grid_coll = collections.defaultdict(list)
        [grid_coll[out_ids[i]].append(img) for i, img in enumerate(samples)]
        for k, v in grid_coll.items():
            v = np.stack(v).reshape((2, 2, -1, args.width, 3)).transpose((0, 2, 1, 3, 4))
            out_img_file = os.path.join(args.outdir, k + ".png")
            PIL.Image.fromarray(v.reshape((-1, 2 * args.width, 3))).save(out_img_file)
    (dist.barrier(device_ids=[rank]), dist.destroy_process_group()) if world_size > 1 else None
