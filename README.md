<div align="center">

<h1>🐻 URSA: Uniform Discrete Diffusion with Metric Path<br>for Video Generation</h1>

<p align="center">
<a href="https://arxiv.org/abs/2510.24717"><img src="https://img.shields.io/badge/ArXiv-2510.24717-%23840707.svg" alt="ArXiv"></a>
<a href="https://huggingface.co/spaces/BAAI/nova-d48w1024-osp480"><img src="https://img.shields.io/badge/🤗 Demo-TI2V-%26840707.svg" alt="T2VDemo"></a>
<a href="http://bitterdhg.github.io/URSA_page"><img src="https://img.shields.io/badge/Project-URSA-%237CB4F7.svg" alt="Project"></a>
</p>

<p align="center">

[Haoge Deng](https://scholar.google.com/citations?user=S2sbvjgAAAAJ&hl)<sup>1,4*</sup>, [Ting Pan](https://scholar.google.com/citations?&user=qQv6YbsAAAAJ)<sup>2,4*</sup>, [Fan Zhang](https://scholar.google.com/citations?user=VsJ39HMAAAAJ)<sup>4*</sup>, [Yang Liu](https://scholar.google.com/citations?user=9JcQ2hwAAAAJ&hl)<sup>3,4*</sup>, [Zhuoyan Luo](https://scholar.google.com/citations?user=mKQhEsIAAAAJ&hl)<sup>4</sup>, [Yufeng Cui](https://scholar.google.com/citations?user=5Ydha2EAAAAJ&hl)<sup>4</sup>, [Wenxuan Wang](https://scholar.google.com/citations?user=75OyC-oAAAAJ&hl)<sup>4</sup><br>
[Chunhua Shen](https://scholar.google.com/citations?user=Ljk2BvIAAAAJ&hl)<sup>3</sup>, [Shiguang Shan](https://scholar.google.com/citations?user=Vkzd7MIAAAAJ&hl)<sup>2</sup>, [Zhaoxiang Zhang](https://scholar.google.com/citations?user=qxWfV6cAAAAJ&hl)<sup>1†</sup>, [Xinlong Wang](https://scholar.google.com/citations?user=DPz0DjYAAAAJ&hl)<sup>4†</sup><br>

[CASIA](http://english.ia.cas.cn)<sup>1</sup>, [CASICT](http://english.ict.cas.cn)<sup>2</sup>, [ZJU](https://www.zju.edu.cn/english)<sup>3</sup>, [BAAI](https://www.baai.ac.cn/en)<sup>4</sup><br>
<sup>*</sup> Equal Contribution, <sup>†</sup> Corresponding Author
<br><br><image src="assets/model_overview.jpg"/>
</div>

We present **URSA** (**U**niform disc**R**ete diffu**S**ion with metric p**A**th), a simple yet powerful framework that bridges the gap with continuous approaches. **URSA** formulates the video generation task as an iterative global refinement of discrete spatiotemporal tokens and scales efficiently to long video generation, requiring fewer inference steps. **URSA** enables multi-task video generation with asynchronous timestep scheduling strategy in one unified model.

## 🚀 News
- ```[Oct 2025]``` Released <a href="https://huggingface.co/spaces/BAAI/nova-d48w1024-osp480"><b>TI2V</b></a> 🤗 Demo.
- ```[Oct 2025]``` Released [Paper](https://arxiv.org/abs/2510.24717) & [Project Page](http://bitterdhg.github.io/URSA_page) & [Evaluation Guide](./docs/evaluation.md).

## ✨Hightlights

- 🥇 **Novel Approach**: Uniform Discrete Diffusion with Metric Path.
- 🥈 **SOTA Performance**: High efficiency with state-of-the-art T2I/T2V/I2V results.
- 🥉 **Unified Modeling**: Multi-task capabilities in a single unified model.

## 🗄️ Models

### 🖼️ Text to Image

| Model | Resolution | Data | Weight | GenEval | DPGBench |
|:-----:|:----------:|:----:|:------:|:-------:|:--------:|
| URSA-1.7B-IBQ1024 | 1024x1024 | 30M | [🤗 HF link](https://huggingface.co/BAAI/URSA-1.7B-IBQ1024) | 0.80 | 86.0 |

### 🎬 Text to Video

| Model | Resolution | Data | Weight | VBench-T2V | VBench-I2V |
|:-----:|:----------:|:----:|:-------|:----------:|:----------:|
| URSA-1.7B-FSQ320 | 49x512x320 | 24M | [🤗 HF link](https://huggingface.co/BAAI/URSA-1.7B-FSQ320) | 82.4 | 86.2 |

## 📖 Table of Contents
- [🔧 Installation](#installation)
- [🔥 Quick Start](#quick-start)
  - [🖼️ Image Generation](#quickstart-image-generation)
  - [🎬 Video Generation](#quickstart-video-generation)
- [💻 Gradio Demo](#gradio-demo)
- [💯 Evaluation](./docs/evaluation.md)

## 🔧 Installation
<a id="installation"></a>

Clone this repository to local disk and install:
```bash
pip install diffusers transformers accelerate imageio imageio-ffmpeg omegaconf wandb
git clone https://github.com/baaivision/URSA.git
cd URSA && pip install .
```

## 🔥 Quick Start
<a id="quick-start"></a>

### 🖼️ Image Generation
<a id="quickstart-image-generation"></a>

```python
import torch
from diffnext.pipelines import URSAPipeline

model_id, height, width = "BAAI/URSA-1.7B-IBQ1024", 1024, 1024
model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
pipe = URSAPipeline.from_pretrained(model_id, **model_args)
pipe = pipe.to(torch.device("cuda"))

prompt = "The bear, calm and still, gazes upward as if lost in contemplation of the cosmos."
negative_prompt = "worst quality, low quality, inconsistent motion, static, still, blurry, jittery, distorted, ugly"

image = pipe(**locals()).frames[0]
image.save("ursa.jpg")
```

### 🎬 Video Generation
<a id="quickstart-video-generation"></a>

```python
import os, torch, numpy
from diffnext.pipelines import URSAPipeline
from diffnext.utils import export_to_video
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_id, height, width = "BAAI/URSA-1.7B-FSQ320", 320, 512
model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
pipe = URSAPipeline.from_pretrained(model_id, **model_args)
pipe = pipe.to(torch.device("cuda"))

text_prompt = "a lone grizzly bear walks through a misty forest at dawn, sunlight catching its fur."
negative_prompt = "worst quality, low quality, inconsistent motion, static, still, blurry, jittery, distorted, ugly"

# Text-to-Image
prompt = text_prompt
num_frames, num_inference_steps = 1, 25
image = pipe(**locals()).frames[0]
image.save("ursa.jpg")

# Image-to-Video
prompt = f"motion=9.0, {text_prompt}"
num_frames, num_inference_steps = 49, 50
video = pipe(**locals()).frames[0]
export_to_video(video, "ursa_1+48f.mp4", fps=12)

# Text-to-Video
image, video = None, None
prompt = f"motion=9.0, {text_prompt}"
num_frames, num_inference_steps = 49, 50
video = pipe(**locals()).frames[0]
export_to_video(video, "ursa_49f.mp4", fps=12)

# Video-to-Video
prompt = f"motion=5.0, {text_prompt}"
num_frames, num_inference_steps = 49, 50
num_cond_frames, cond_noise_scale = 13, 0.1
for i in range(12):
    video, start_video = video[-num_cond_frames:], video
    video = pipe(**locals()).frames[0]
    video = numpy.concatenate([start_video, video[num_cond_frames:]])
    export_to_video(video, "ursa_{}f.mp4".format(video.shape[0]), fps=12)
```

## 💻 Gradio Demo
<a id="gradio-demo"></a>

```bash
# Text-to-Image (T2I)
python scripts/app_ursa_t2i.py --model "BAAI/URSA-1.7B-IBQ1024" --device 0

# Text-to-Image-to-Video (TI2V)
python scripts/app_ursa_ti2v.py --model "BAAI/URSA-1.7B-FSQ320" --device 0
```

## 📋 Todo List
- [X] [Model Zoo](#model-zoo)
- [X] [Quick Start](#quick-start)
- [X] [Gradio Demo](#gradio-demo)
- [X] [Evaluation Guide](./docs/evaluation.md)
- [ ] [Training Guide](./docs/training.md)
- [ ] 0.6B and 4B Model

## 📖 Citation
If you find this repository useful, please consider giving a star ⭐ and citation 🦖:
```
@article{deng2025ursa,
  title={Uniform Discrete Diffusion with Metric Path for Video Generation},
  author={Deng, Haoge and Pan, Ting and Zhang, Fan and Liu, Yang and Luo, Zhuoyan and Cui, Yufeng and Shen, Chunhua and Shan, Shiguang and Zhang, Zhaoxiang and Wang, Xinlong},
  journal={arXiv preprint arXiv:2510.24717},
  year={2025}
}
```
```
@article{deng2024nova,
  title={Autoregressive Video Generation without Vector Quantization},
  author={Deng, Haoge and Pan, Ting and Diao, Haiwen and Luo, Zhengxiong and Cui, Yufeng and Lu, Huchuan and Shan, Shiguang and Qi, Yonggang and Wang, Xinlong},
  journal={arXiv preprint arXiv:2412.14169},
  year={2024}
}
```

## 🤗 Acknowledgement

We thank the repositories: 
- [NOVA](https://github.com/baaivision/NOVA). ✨NOVA is the predecessor of 🐻URSA.
- [FlowMatching](https://github.com/facebookresearch/flow_matching). This codebase systemically provides CFM and DFM implementations.
- [FUDOKI](https://github.com/fudoki-hku/FUDOKI). This codebase provides a naive multimodal DFM implementation.
- [CodeWithGPU](https://github.com/seetacloud/codewithgpu). CodeWithGPU library is the core of our data loading pipeline.

## License
Code and models are licensed under [Apache License 2.0](LICENSE).
