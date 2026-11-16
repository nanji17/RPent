# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Run native Molmo2 pointing locally on Apple Silicon using quantized MLX weights."""

import io

from PIL import Image

from rpent.robots.components.molmo_client import MolmoResult, identity_prompt
from rpent.robots.components.molmo_server import _parse_point


class MlxMolmoGrounder:
    """A local VisualGrounder with the same native prompt and normalized-coordinate contract."""

    def __init__(self, checkpoint):
        from mlx_vlm import load

        self.model, self.processor = load(str(checkpoint))

    def _generate(self, raster, instruction, tokens, *, skip_special_tokens=False):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(
            self.processor,
            self.model.config,
            instruction,
            num_images=1,
        )
        result = generate(
            self.model,
            self.processor,
            prompt,
            image=[raster],
            max_tokens=tokens,
            temperature=0.0,
            skip_special_tokens=skip_special_tokens,
            verbose=False,
        )
        return result.text

    def verify(self, image: bytes, query: str | None, *, isolated: bool = False) -> str:
        raster = Image.open(io.BytesIO(image)).convert("RGB")
        return self._generate(
            raster,
            identity_prompt(query, isolated=isolated),
            32 if query is None and not isolated else 20 if query is None else 8,
            skip_special_tokens=True,
        )

    def ground(self, image: bytes, query: str) -> MolmoResult:
        raster = Image.open(io.BytesIO(image)).convert("RGB")
        answer = self._generate(raster, f"Point to {query.rstrip('.')}.", 48)
        normalized = _parse_point(answer, require_single=True)
        width, height = raster.size
        xy = (
            (
                min(width - 1, normalized[0] * width / 1000),
                min(height - 1, normalized[1] * height / 1000),
            )
            if normalized is not None
            else None
        )
        return MolmoResult(
            found=xy is not None,
            point_xy=xy,
            image_size=raster.size,
            answer=answer,
        )
