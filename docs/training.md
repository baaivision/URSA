# Training Guide
This guide provides simple snippets to train diffnext models.

# 1. Build VQVAE cache
To optimize training workflow, we preprocess images or videos into VQVAE latents.

## Requirements:
```bash
pip install protobuf==3.20.3 codewithgpu decord
```

## Build T2I cache
Following snippet can be used to cache image latents:

```python
import os, codewithgpu, torch, PIL.Image, numpy as np
from diffnext.models.autoencoders.autoencoder_vq import AutoencoderVQ

device, dtype = torch.device("cuda"), torch.float16
vae = AutoencoderVQ.from_pretrained("/path/to/BAAI/URSA-1.7B-IBQ1024/vae")
vae = vae.to(device=device, dtype=dtype).eval()

features = {"codes": "bytes", "caption": "string", "text": "string", "shape": ["int64"]}
_, writer = os.makedirs("./img_dataset"), codewithgpu.RecordWriter("./img_dataset", features)

img = PIL.Image.open("./assets/sample_image.jpg")
x = torch.as_tensor(np.array(img)[None, ...].transpose(0, 3, 1, 2)).to(device).to(dtype)
with torch.no_grad():
    x = vae.encode(x.sub(127.5).div(127.5)).latent_dist.parameters.cpu().numpy()[0]
example = {"caption": "long caption", "text": "short text"}
# Ensure enough examples for codewithgou distributed dataset.
[writer.write({"shape": x.shape, "codes": x.tobytes(), **example}) for _ in range(16)]
writer.close()
```

## Build T2V cache
Following snippet can be used to cache video latents:

```python
import os, codewithgpu, torch, decord, numpy as np
from diffnext.models.autoencoders.autoencoder_vq_cosmos3d import AutoencoderVQCosmos3D

device, dtype = torch.device("cuda"), torch.float16
vae = AutoencoderVQCosmos3D.from_pretrained("/path/to/URSA-1.7B-FSQ320/vae")
vae = vae.to(device=device, dtype=dtype).eval()

features = {"codes": "bytes", "caption": "string", "text": "string", "shape": ["int64"], "flow": "float64"}
_, writer = os.makedirs("./vid_dataset"), codewithgpu.RecordWriter("./vid_dataset", features)

resize, crop_size, frame_ids = 320, (320, 512), list(range(0, 97, 2))
vid = decord.VideoReader("./assets/sample_video.mp4")
h, w = vid[0].shape[:2]
scale = float(resize) / float(min(h, w))
size = int(h * scale + 0.5), int(w * scale + 0.5)
y, x = (size[0] - crop_size[0]) // 2, (size[1] - crop_size[1]) // 2
vid = decord.VideoReader("./assets/sample_video.mp4", height=size[0], width=size[1])
vid = vid.get_batch(frame_ids).asnumpy()
vid = vid[:, y : y + crop_size[0], x : x + crop_size[1]]
x = torch.as_tensor(vid[None, ...].transpose((0, 4, 1, 2, 3))).to(device).to(dtype)
with torch.no_grad():
    x = vae.encode(x.sub(127.5).div(127.5)).latent_dist.parameters.cpu().numpy()[0]
example = {"caption": "long caption", "text": "short text", "flow": 9}
[writer.write({"shape": x.shape, "codes": x.tobytes(), **example}) for _ in range(16)]
writer.close()
```
