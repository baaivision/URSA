# Evaluations

## GenEval

### 1. Sample prompt images
```bash
python ./evaluations/geneval/sample.py \
--height 1024 --width 1024 \
--guidance_scale 7 --num_inference_steps 25 \
--ckpt /path/to/URSA-1.7B-IBQ1024 \
--prompt_size 4 --outdir ./samples/geneval/URSA-1.7B-IBQ1024
```

### 2. Evaluation
<IMAGE_FOLDER>=./samples/geneval/URSA-1.7B-IBQ1024

Please refer [GenEval](https://github.com/djghosh13/geneval?tab=readme-ov-file#evaluation) evaluation guide.

## DPG-Bench

### 1. Sample prompt images
```bash
python evaluations/dpgbench/sample.py \
--height 1024 --width 1024 \
--guidance_scale 7 --num_inference_steps 25 \
--ckpt ./checkpoints/URSA-1.7B-IBQ1024 \
--prompt_size 4 --outdir samples/dpgbench/URSA-1.7B-IBQ1024
```

### 2. Evaluation
<IMAGE_FOLDER>=./samples/dpgbench/URSA-1.7B-IBQ1024

Please refer [DPG-Bench](https://github.com/TencentQQGYLab/ELLA?tab=readme-ov-file#-dpg-bench) evaluation guide.

## VBench

### 1. Sample prompt videos
```bash
python evaluations/vbench/sample.py \
--num_frames 49 --height 320 --width 512 \
--guidance_scale 7 --num_inference_steps 50 --motion_score 9 \
--ckpt ./checkpoints/URSA-1.7B-FSQ320 \
--prompt_size 1 --outdir ./samples/vbench/URSA-1.7B-FSQ320
```

### 2. Evaluation
<VIDEO_FOLDER>=./samples/vbench/URSA-1.7B-FSQ320

Please refer [VBench](https://github.com/Vchitect/VBench?tab=readme-ov-file#evaluation-on-the-standard-prompt-suite-of-vbench) evaluation guide.
