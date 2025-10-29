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
"""URSA TI2V application."""

import argparse
import os

import gradio as gr
import numpy as np
import PIL.Image
import torch

from diffnext.pipelines import URSAPipeline
from diffnext.utils import export_to_image, export_to_video

# Fix tokenizer fork issue.
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Switch to the allocator optimized for dynamic shape.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Serve URSA TI2V application")
    parser.add_argument("--model", default="", help="model path")
    parser.add_argument("--device", type=int, default=0, help="device index")
    parser.add_argument("--precision", default="float16", help="compute precision")
    return parser.parse_args()


def crop_image(image, target_h, target_w):
    """Center crop image to target size."""
    h, w = image.height, image.width
    aspect_ratio_target, aspect_ratio = target_w / target_h, w / h
    if aspect_ratio > aspect_ratio_target:
        new_w = int(h * aspect_ratio_target)
        x_start = (w - new_w) // 2
        image = image.crop((x_start, 0, x_start + new_w, h))
    else:
        new_h = int(w / aspect_ratio_target)
        y_start = (h - new_h) // 2
        image = image.crop((0, y_start, w, y_start + new_h))
    return np.array(image.resize((target_w, target_h), PIL.Image.Resampling.BILINEAR))


def generate_image(
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    guidance_scale,
    num_inference_steps=25,
):
    """Generate a video."""
    args = {**locals(), **video_presets["t2i"]}
    seed = np.random.randint(2147483647) if randomize_seed else seed
    device = getattr(pipe, "_offload_device", pipe.device)
    generator = torch.Generator(device=device).manual_seed(seed)
    images = pipe(generator=generator, **args).frames
    return [export_to_image(image, quality=95) for image in images] + [seed]


def generate_video(
    prompt,
    negative_prompt,
    image,
    motion_score,
    seed,
    randomize_seed,
    guidance_scale,
    num_inference_steps,
    output_type="np",
):
    """Generate a video."""
    args = {**locals(), **video_presets["ti2v"]}
    args["prompt"] = f"motion={motion_score:.1f}, {prompt}"
    args["image"] = crop_image(image, args["height"], args["width"]) if image else None
    seed = np.random.randint(2147483647) if randomize_seed else seed
    device = getattr(pipe, "_offload_device", pipe.device)
    generator = torch.Generator(device=device).manual_seed(seed)
    frames = pipe(generator=generator, **args).frames[0]
    return export_to_video(frames, fps=12), seed


css = """#col-container {margin: 0 auto; max-width: 1366px}"""
title = "Uniform Discrete Diffusion with Metric Path for Video Generation"
header = (
    "<div align='center'>"
    "<h2>Uniform Discrete Diffusion with Metric Path for Video Generation</h2>"
    "<h3><a href='https://arxiv.org/abs/2510.24717' target='_blank' rel='noopener'>[paper]</a>"
    "<a href='https://github.com/baaivision/URSA' target='_blank' rel='noopener'>[code]</a></h3>"
    "</div>"
)

video_presets = {
    "t2i": {"width": 512, "height": 320, "num_frames": 1},
    "ti2v": {"width": 512, "height": 320, "num_frames": 49},
}

prompts = [
    "a lone grizzly bear walks through a misty forest at dawn, sunlight catching its fur.",
    "Many spotted jellyfish pulsating under water. Their bodies are transparent and glowing in deep ocean.",  # noqa
    "An intense close-up of a soldier’s face, covered in dirt and sweat, his eyes filled with determination as he surveys the battlefield.",  # noqa
    "a close-up shot of a woman standing in a dimly lit room. she is wearing a traditional chinese outfit, which includes a red and gold dress with intricate designs and a matching headpiece. the woman has her hair styled in an updo, adorned with a gold accessory. her makeup is done in a way that accentuates her features, with red lipstick and dark eyeshadow. she is looking directly at the camera with a neutral expression. the room has a rustic feel, with wooden beams and a stone wall visible in the background. the lighting in the room is soft and warm, creating a contrast with the woman's vibrant attire. there are no texts or other objects in the video. the style of the video is a portrait, focusing on the woman and her attire.",  # noqa
    "The camera slowly rotates around a massive stack of vintage televisions that are placed within a large New York museum gallery. Each of the televisions is showing a different program. There are 1950s sci-fi movies with their distinctive visuals, horror movies with their creepy scenes, news broadcasts with moving images and words, static on some screens, and a 1970s sitcom with its characteristic look. The televisions are of various sizes and designs, some with rounded edges and others with more angular shapes. The gallery is well-lit, with light falling on the stack of televisions and highlighting the different programs being shown. There are no people visible in the immediate vicinity, only the stack of televisions and the surrounding gallery space.",  # noqa
]
motion_scores = [9, 9, 9, 9, 9]
videos = ["", "", "", "", ""]
examples = [list(x) for x in zip(prompts, motion_scores)]


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.device)
    model_args = {"torch_dtype": getattr(torch, args.precision.lower()), "trust_remote_code": True}
    pipe = URSAPipeline.from_pretrained(args.model, **model_args).to(device)

    # Application.
    app = gr.Blocks(css=css, theme="origin").__enter__()
    container = gr.Column(elem_id="col-container").__enter__()
    _, main_row = gr.Markdown(header), gr.Row().__enter__()

    # Input.
    input_col = gr.Column().__enter__()
    prompt = gr.Text(
        label="Prompt",
        placeholder="Describe the video you want to generate",
        value="A lone grizzly bear walks through a misty forest at dawn, sunlight catching its fur.",  # noqa
        lines=5,
    )
    negative_prompt = gr.Text(
        label="Negative Prompt",
        placeholder="Describe what you don't want in the video",
        value="worst quality, low quality, inconsistent motion, static, still, blurry, jittery, distorted, ugly",  # noqa
        lines=1,
    )
    with gr.Row():
        generate_image_btn = gr.Button("Generate Image Prompt", variant="primary", size="lg")
        generate_video_btn = gr.Button("Generate Video", variant="primary", size="lg")
    image_prompt = gr.Image(label="Image Prompt", height=480, type="pil")

    # fmt: off
    options = gr.Accordion("Options", open=False).__enter__()
    seed = gr.Slider(label="Seed", maximum=2147483647, step=1, value=0)
    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
    guidance_scale = gr.Slider(label="Guidance scale", minimum=1, maximum=10.0, step=0.1, value=7.0)
    with gr.Row():
        num_inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=100, step=1, value=50)  # noqa
    options.__exit__(), input_col.__exit__()

    # Results.
    result_col = gr.Column().__enter__()
    motion = gr.Slider(label="Motion Score", minimum=1, maximum=10, step=1, value=9)
    result = gr.Video(label="Result", height=480, show_label=False, autoplay=True)
    result_col.__exit__(), main_row.__exit__()
    # fmt: on

    # Examples.
    with gr.Row():
        gr.Examples(examples=examples, inputs=[prompt, motion])

    # Events.
    container.__exit__()
    gr.on(
        triggers=[generate_image_btn.click, prompt.submit, negative_prompt.submit],
        fn=generate_image,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            guidance_scale,
        ],
        outputs=[image_prompt, seed],
    )
    gr.on(
        triggers=[generate_video_btn.click, prompt.submit, negative_prompt.submit],
        fn=generate_video,
        inputs=[
            prompt,
            negative_prompt,
            image_prompt,
            motion,
            seed,
            randomize_seed,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[result, seed],
    )
    app.__exit__(), app.launch(share=False)
