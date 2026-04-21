# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Image rewards."""

from io import BytesIO
import pickle
import requests
from requests.adapters import HTTPAdapter, Retry
from typing import Dict, List, Tuple
from typing_extensions import Self

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as vision_funcs
from transformers import CLIPProcessor, CLIPModel


def resize_longest_edge(img, max_size=224, fill=0):
    size = tuple(round(dim * max_size / max(img.size)) for dim in img.size[::-1])
    pad_h, pad_w = [max_size - side for side in size]
    padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
    return vision_funcs.pad(vision_funcs.resize(img, size, interpolation=3), padding, fill)


class BaseScorer(object):
    """Base scorer."""

    def __init__(self, batch_size=8, address=None, port=None, **kwargs):
        self.batch_size, self.model = batch_size, None
        self.url = f"{address}:{port}" if address else None
        self.sess = requests.Session() if address else None
        retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
        self.sess.mount("http://", HTTPAdapter(max_retries=retries)) if address else None

    @property
    def dtype(self) -> torch.dtype:
        """Return the execution dtype."""
        return list(self.model.parameters())[0].dtype if self.model else None

    @property
    def device(self) -> torch.device:
        """Return the execution device."""
        return list(self.model.parameters())[0].device if self.model else None

    def to(self, *args, **kwargs) -> Self:
        """Convert model according to given arguments."""
        return (self.model.to(*args, **kwargs) if self.model else None, self)[1]

    @staticmethod
    def image_to_bytes(image, image_format="JPEG") -> bytes:
        """Convert image to bytes."""
        buffer = BytesIO()
        PIL.Image.fromarray(image).save(buffer, format=image_format)
        return buffer.getvalue()

    def batch_iter(self, input) -> List:
        """Return a batch iterator."""
        num_slices = int(np.ceil(len(input) / self.batch_size))
        k, m = divmod(len(input), num_slices)
        slices = [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(num_slices)]
        return [input[slice(*s)] for s in slices]

    def post(self, data_dict: Dict) -> Dict:
        """Post data dict."""
        data_bytes = pickle.dumps(data_dict)
        response = self.sess.post(self.url, data=data_bytes, timeout=120)
        return pickle.loads(response.content)


class GenEvalScorer(BaseScorer):
    """GenEval scorer."""

    def __init__(self, batch_size=8, **kwargs):
        super().__init__(batch_size, **kwargs)
        # See https://github.com/Yovecent/UDM-GRPO
        self.model_path = "/path/to/mmdetection"
        self.model, self.initialized = torch.nn.Linear(1, 1), 0  # Dummy

    def __call__(self, images, prompts, metadatas) -> Tuple[List[float], List[float]]:
        from diffnext.rewards.evaluate_geneval import create_model, compute_score

        if not self.initialized:
            self.initialized, _ = 1, create_model(self.model_path, self.device)
        if isinstance(images, torch.Tensor):
            images = images.mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            images = [PIL.Image.fromarray(image) for image in images]
        scores, rewards = [], []
        for x, y in zip(self.batch_iter(images), self.batch_iter(metadatas)):
            _, __ = compute_score(list(x), list(y))[:2]
            scores.extend(_), rewards.extend(__)
        return scores, rewards


class GenEvalRemoteScorer(BaseScorer):
    """GenEval remote scorer."""

    def __init__(self, batch_size=8, address="http://127.0.0.1", port=18085, **kwargs):
        super().__init__(batch_size, address, port, **kwargs)

    def __call__(self, images, prompts, metadatas) -> Tuple[List[float], List[float]]:
        if isinstance(images, torch.Tensor):
            images = images.mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        scores, rewards = [], []
        for x, y in zip(self.batch_iter(images), self.batch_iter(metadatas)):
            data = {"images": [self.image_to_bytes(_) for _ in x], "metadatas": list(y)}
            response = self.post(data)
            scores.extend(response["scores"]), rewards.extend(response["rewards"])
        return scores, rewards


class CLIPScorer(BaseScorer):
    """CLIP scorer."""

    def __init__(self, batch_size=8, **kwargs):
        super().__init__(batch_size, **kwargs)
        model_path = "openai/clip-vit-large-patch14"
        self.processor = CLIPProcessor.from_pretrained(model_path, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_path, torch_dtype=torch.float16).eval()
        self.tokenizer_args = {"padding": True, "truncation": True, "max_length": 77}
        self.image_transform, self.max_score = lambda x: x, 100.0

    @torch.no_grad()
    def compute_score(self, images, prompts) -> torch.Tensor:
        device, dtype = self.device, self.dtype
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device).to(dtype) for k, v in inputs.items()}
        image_embs = torch.nn.functional.normalize(self.model.get_image_features(**inputs), dim=-1)
        inputs = self.processor(text=prompts, **self.tokenizer_args, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_embs = torch.nn.functional.normalize(self.model.get_text_features(**inputs), dim=-1)
        logit_scale = self.model.logit_scale.float().exp()
        return logit_scale * (text_embs @ image_embs.T).float().diag()

    def __call__(self, images, prompts, metadatas=None) -> Tuple[List[float], List[float]]:
        if isinstance(images, torch.Tensor):
            images = images.mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            images = [PIL.Image.fromarray(image) for image in images]
        images, scores, rewards = [self.image_transform(image) for image in images], [], []
        for x, y in zip(self.batch_iter(images), self.batch_iter(prompts)):
            _ = self.compute_score(x, y)
            scores.extend(_.div(self.max_score).tolist()), rewards.extend(_.tolist())
        return scores, rewards


class PickScorer(CLIPScorer):
    """PickScore scorer."""

    def __init__(self, batch_size=8, **kwargs):
        BaseScorer.__init__(self, batch_size, **kwargs)
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = CLIPProcessor.from_pretrained(model_path, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_path, torch_dtype=torch.float16).eval()
        self.tokenizer_args = {"padding": True, "truncation": True, "max_length": 77}
        self.image_transform, self.max_score = lambda x: x, 26.0


class HPSv2Scorer(CLIPScorer):
    """HPSv2 scorer."""

    def __init__(self, batch_size=8, **kwargs):
        BaseScorer.__init__(self, batch_size, **kwargs)
        model_path = "xswu/HPSv2"
        self.processor = CLIPProcessor.from_pretrained(model_path, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_path, torch_dtype=torch.float16).eval()
        self.tokenizer_args = {"padding": True, "truncation": True, "max_length": 77}
        self.image_transform, self.max_score = resize_longest_edge, 100.0


class OCRScorer(BaseScorer):
    """OCR scorer."""

    def __init__(self, batch_size=8, **kwargs):
        super().__init__(batch_size, **kwargs)
        from paddleocr import PaddleOCR

        model_path = "paddleocr/models"
        args = {"use_angle_cls": False, "lang": "en", "show_log": False, "use_gpu": False}
        args.update({"det_model_dir": f"{model_path}/det", "rec_model_dir": f"{model_path}/rec"})
        self.model = PaddleOCR(cls_model_dir=f"{model_path}/cls", **args)

    def to(self, *args, **kwargs) -> Self:
        return self

    @torch.no_grad()
    def compute_score(self, images, prompts) -> List[float]:
        from Levenshtein import distance

        prompts, rewards = [prompt.split('"')[1] for prompt in prompts], []
        for img, prompt in zip(images, prompts):
            try:
                result = self.model.ocr(img, cls=False)
                recognized_text = (
                    "".join([res[1][0] if res[1][1] > 0 else "" for res in result[0]])
                    if result[0]
                    else ""
                )
                recognized_text = recognized_text.replace(" ", "").lower()
                prompt = prompt.replace(" ", "").lower()
                dist = 0 if prompt in recognized_text else distance(recognized_text, prompt)
                dist = min(dist, len(prompt))  # only add one character penalty
            except Exception:
                dist = len(prompt)  # Maximum penalty
            rewards.append(1 - dist / len(prompt))
        return rewards

    def __call__(self, images, prompts, metadatas=None) -> Tuple[List[float], List[float]]:
        if isinstance(images, torch.Tensor):
            images = images.mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        scores, rewards = [], []
        for x, y in zip(self.batch_iter(images), self.batch_iter(prompts)):
            _ = self.compute_score(x, y)
            scores.extend(_), rewards.extend(_)
        return scores, rewards


class ComposeScorer(object):
    """Compose scorer."""

    SCORERS = {
        "ocr": OCRScorer,
        "hpsv2": HPSv2Scorer,
        "clipscore": CLIPScorer,
        "pickscore": PickScorer,
        "geneval": GenEvalScorer,
        "geneval_remote": GenEvalRemoteScorer,
    }
    SCORER_NAMES = dict((v, k) for k, v in SCORERS.items())

    def __init__(self, scorers, weights=None, **kwargs):
        self.scorers = [self.SCORERS[_](**kwargs) if isinstance(_, str) else _ for _ in scorers]
        self.weights = weights if weights else [1.0] * len(scorers)

    def to(self, *args, **kwargs) -> Self:
        return ([scorer.to(*args, **kwargs) for scorer in self.scorers], self)[1]

    def __call__(self, images, prompts, metadatas=None) -> Tuple[List[float], List[float]]:
        scores, rewards = [], {}
        for scorer, weight in zip(self.scorers, self.weights):
            score, rewards[self.SCORER_NAMES[type(scorer)]] = scorer(images, prompts, metadatas)
            scores.append(np.array(score) * weight)
        return sum(scores).tolist(), rewards
