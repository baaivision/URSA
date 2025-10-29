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
"""VBench sampling for URSA models."""

import argparse
import collections
import os
import os.path as osp

import imageio
from tqdm import tqdm
import torch
import torch.distributed as dist

from diffnext.pipelines import URSAPipeline


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="vbench sampling")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint file")
    parser.add_argument("--prompt", type=str, default=None, help="prompt folder")
    parser.add_argument("--num_frames", type=int, default=49, help="number of frames")
    parser.add_argument("--height", type=int, default=320, help="video height")
    parser.add_argument("--width", type=int, default=512, help="video width")
    parser.add_argument("--motion_score", type=float, default=9.0, help="motion score")
    parser.add_argument("--guidance_scale", type=float, default=7, help="guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="inference steps")
    parser.add_argument("--prompt_size", type=int, default=1, help="prompt size for each batch")
    parser.add_argument("--sample_size", type=int, default=5, help="sample size for each prompt")
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
    args.prompt = args.prompt if args.prompt else osp.join(osp.dirname(__file__), "prompts")
    prompt_files = [osp.join(args.prompt, x) for x in os.listdir(args.prompt) if x.endswith(".txt")]
    prompt_files.sort()
    dense_prompt_files = [x for x in prompt_files if "longer" in x]
    raw_prompt_files = [x for x in prompt_files if "longer" not in x]
    raw_prompts, dense_prompts, names = [], [], []
    for raw_prompt_file, dense_prompt_file in zip(raw_prompt_files, dense_prompt_files):
        track = raw_prompt_file.split("/")[-1].replace(".txt", "")
        raw_prompts += [_.strip() for _ in open(raw_prompt_file).readlines()]
        names += [f"{args.outdir}/{track}/{txt}.mp4" for txt in raw_prompts[len(dense_prompts) :]]
        dense_prompts += [_.strip() for _ in open(dense_prompt_file).readlines()]
    txts, caps, names = map(lambda _: _[rank::world_size], (raw_prompts, dense_prompts, names))

    # Arguments.
    gen_args = {"guidance_scale": args.guidance_scale}
    gen_args["vae_batch_size"] = args.vae_batch_size
    gen_args["num_inference_steps"] = args.num_inference_steps
    gen_args["height"], gen_args["width"] = args.height, args.width
    gen_args["num_frames"] = args.num_frames
    negative_prompt = "worst quality, low quality, inconsistent motion, static, still, blurry, jittery, distorted, ugly"  # noqa
    gen_args["negative_prompt"] = negative_prompt

    # Pipeline.
    pipe = URSAPipeline.from_pretrained(args.ckpt, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    for step in tqdm(range(0, len(dense_prompts), args.prompt_size), disable=rank):
        samples, paths, gen_args["generator"] = [], [], generator
        prompts = dense_prompts[step : step + args.prompt_size]
        prompts = [f"motion={args.motion_score:.1f}, {text_prompt}" for text_prompt in prompts]
        paths = names[step : step + args.prompt_size] * args.sample_size
        [samples.extend(pipe(prompts, **gen_args).frames) for _ in range(args.sample_size)]
        name_cnt = collections.defaultdict(int)
        for idx, frames in enumerate(samples):
            name = paths[idx].replace(".mp4", "-{}.mp4".format(name_cnt[paths[idx]]))
            os.makedirs(os.path.dirname(name), exist_ok=True)
            name_cnt[paths[idx]] += 1
            with imageio.get_writer(name, fps=12, ffmpeg_log_level="error") as writer:
                [writer.append_data(frames[k]) for k in range(frames.shape[0])]
    (dist.barrier(device_ids=[rank % 8]), dist.destroy_process_group()) if world_size > 1 else None
