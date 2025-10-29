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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, esither express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Export utilities."""

import tempfile

try:
    import imageio
except ImportError:
    imageio = None
import PIL.Image


def export_to_image(image, output_image_path=None, suffix=".webp", quality=100):
    """Export to image."""
    if output_image_path is None:
        output_image_path = tempfile.NamedTemporaryFile(suffix=suffix).name
    if isinstance(image, PIL.Image.Image):
        image.save(output_image_path, quality=quality)
    else:
        PIL.Image.fromarray(image).save(output_image_path, quality=quality)
    return output_image_path


def export_to_video(video_frames, output_video_path=None, fps=12):
    """Export to video."""
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    if imageio is None:
        raise ImportError("Failed to import <imageio> library.")
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)
    return output_video_path
