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

"""Evaluate a staged, local Molmo + Pydantic AI + SAM3 pipeline on verified RGB-D frames."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from rpent.robots.components.molmo_client import MolmoResult
from rpent.robots.components.sam3_client import Sam3Result
from rpent.session import EnvState
from toolkits.geometry.evaluation.checklist import build_manifest
from toolkits.geometry.evaluation.experiment import prediction_base
from toolkits.geometry.evaluation.libero import (
    FrozenObservationProvider,
    load_observation,
)
from toolkits.geometry.evaluation.replay import ReplayGrounder
from toolkits.geometry.geometry_tool import GeometryTool
from toolkits.geometry.local_point_agent import LocalPointAgent


class RecordedObject:
    """Serve real model responses only for their exact input RGB, point and description."""

    def __init__(self, rgb, query, row, *, identities=None):
        if row is None or row["query"] != query:
            raise ValueError("no recorded evidence for this query")
        if hashlib.sha256(rgb).hexdigest() != row["rgb_sha"]:
            raise ValueError("source PNG hash mismatch")
        self.pixels = hashlib.sha256(
            Image.open(io.BytesIO(rgb)).convert("RGB").tobytes()
        ).digest()
        self.row = row
        self.identities = identities
        self.xy = json.loads(row["xy"])
        keys = row.keys()
        model_query = (
            row["model_query"]
            if "model_query" in keys and row["model_query"]
            else query
        )
        self.prompt = (
            row["model_prompt"]
            if "model_prompt" in keys
            else f"Point to {model_query.rstrip('.')}."
        )

    def _verify(self, image):
        rgb = Image.open(io.BytesIO(image)).convert("RGB")
        if hashlib.sha256(rgb.tobytes()).digest() != self.pixels:
            raise ValueError("requested raster differs from model input")
        return rgb.size

    def ground(self, image, query):
        size = self._verify(image)
        if query != self.row["query"]:
            raise ValueError("requested description differs from model input")
        return MolmoResult(
            self.xy is not None,
            tuple(self.xy) if self.xy else None,
            size,
            self.row["answer"],
        )

    def verify(self, image, query, *, isolated=False):
        return (
            self.identities.verify(image, query, isolated=isolated)
            if self.identities is not None
            else None
        )

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        self._verify(image)
        if (
            min_score != 0.2
            or (point is not None and point != [round(self.xy[1]), round(self.xy[0])])
            or (text_prompt and text_prompt != self.row["query"])
        ):
            raise ValueError("wrong recorded segmentation prompt")
        method = "text" if text_prompt else "point"
        if self.row["method"] is None:
            raise ValueError("segmentation stage is incomplete")
        if method != self.row["method"]:
            return Sam3Result(False, reason="no recorded anchor-consistent mask")
        mask = np.asarray(Image.open(io.BytesIO(self.row["mask"]))) > 0
        return Sam3Result(True, self.row["score"], mask=mask, mask_shape=mask.shape)


def recorded_confirmation(rgb, query, row):
    from toolkits.geometry.local_point_agent import LocalPointAgent

    evidence = RecordedObject(rgb, query, row)
    if "trace" not in row.keys() or row["trace"] is None:
        raise ValueError("confirmation must contain evidence of a self-selected crop")
    trace = json.loads(row["trace"])
    agent = LocalPointAgent(evidence)

    def confirm(image, description, initial):
        if list(initial.point_xy) != trace["initial_xy"]:
            raise ValueError(
                "confirmation evidence refers to a different initial point"
            )
        return agent.ground(image, description)

    return confirm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--identity-replay",
        type=Path,
        help="Directory of exact candidate-verification calls from --save-evidence",
    )
    parser.add_argument("--split", choices=("dev", "validation", "all"), default="all")
    parser.add_argument(
        "--request-ids",
        type=Path,
        help="Optional JSON array of checklist request IDs for matched subsets",
    )
    parser.add_argument(
        "--confirm-with",
        type=Path,
        help="Second local points to consult only when adjacent depth components compete",
    )
    parser.add_argument(
        "--region", choices=("sam", "depth", "hybrid"), default="hybrid"
    )
    parser.add_argument(
        "--extra-features",
        action="store_true",
        help="Also measure dimensional and directional features on the same frame",
    )
    args = parser.parse_args()
    identities = ReplayGrounder(args.identity_replay) if args.identity_replay else None
    db = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    confirmation = None
    if args.confirm_with:
        confirmation = sqlite3.connect(
            f"file:{args.confirm_with.resolve()}?mode=ro", uri=True
        )
        confirmation.row_factory = sqlite3.Row
    manifest = build_manifest()
    selected = None
    if args.request_ids:
        specified = json.loads(args.request_ids.read_text())
        if not isinstance(specified, list) or not all(
            isinstance(k, str) for k in specified
        ):
            parser.error("--request-ids expects an array of strings")
        selected = set(specified)
        known = {
            r["request_id"] for case in manifest["cases"] for r in case["requests"]
        }
        if not selected or selected - known or len(selected) != len(specified):
            parser.error("--request-ids contains missing, repeated or unknown entries")
    with (
        args.predictions.open("x") as dest,
        TemporaryDirectory(prefix="geometry-rgbd-") as tmp,
    ):
        state = EnvState(Path(tmp))
        with state.record_step(state={}):
            pass
        for case in manifest["cases"]:
            if (args.split == "dev" and case["seed"] > 3) or (
                args.split == "validation" and case["seed"] <= 3
            ):
                continue
            requests = [
                r
                for r in case["requests"]
                if selected is None or r["request_id"] in selected
            ]
            if not requests:
                continue
            folder = args.inputs / case["case_id"]
            obs = load_observation(folder, manifest, case)
            source = (folder / "episode/agentview_high.png/00.png").read_bytes()
            for r in requests:
                row = prediction_base(manifest, case, r)
                evidence = RecordedObject(
                    source,
                    row["query"],
                    db.execute(
                        "SELECT * FROM visual WHERE id=?", (r["request_id"],)
                    ).fetchone(),
                    identities=identities,
                )
                tool = GeometryTool(
                    state=state,
                    observations=FrozenObservationProvider(obs),
                    grounder=LocalPointAgent(evidence),
                    segmenter=evidence,
                    object_first=True,
                    region_source=args.region,
                    point_refiner=recorded_confirmation(
                        source,
                        row["query"],
                        confirmation.execute(
                            "SELECT * FROM visual WHERE id=?", (r["request_id"],)
                        ).fetchone(),
                    )
                    if confirmation is not None
                    else None,
                    model="Molmo2-8B/local",
                )
                # Use the published declaration's validation and bound public handler.
                features = [row["quantity"]]
                if args.extra_features:
                    if row["quantity"] == "inner_rim_center":
                        features += [
                            "inner_rim_radius",
                            "rim_extrema",
                            "object_height",
                            "supported_height",
                            "height_above_support",
                        ]
                    elif row["quantity"] == "rectangle_center":
                        features += [
                            "top_size",
                            "top_corners",
                            "cuboid_corners",
                            "object_height",
                            "supported_height",
                            "height_above_support",
                        ]
                args_model = tool.geometry.args_schema.model_validate(
                    {
                        "query": row["query"],
                        "camera": "agentview",
                        "step": 0,
                        "required_geometry": features,
                    },
                    strict=True,
                )
                data = tool.geometry(**args_model.model_dump()).data
                field = data.get("geometry", {}).get(row["quantity"], {})
                primary_status = data["status"]
                if args.extra_features and primary_status != "error":
                    primary_status = "insufficient_evidence"
                    if field.get("value") is not None:
                        primary_status = (
                            "warning" if field.get("warnings") else "usable"
                        )
                if field and (
                    field["world_frame_id"] != row["frame_id"]
                    or field["source_step"] != 0
                    or field["definition"] != row["definition"]
                ):
                    raise ValueError("field semantics differ from evaluation manifest")
                row.update(
                    world_xyz=field.get("value"),
                    visual_prompt=evidence.prompt,
                    fit_accepted=field.get("fit_accepted", False),
                    status=primary_status,
                    warnings=field.get("warnings", []),
                    reason=field.get("reason") or data.get("error"),
                    calls=data["calls"],
                    pixel_rc=data.get("pixel_rc"),
                    initial_pixel_rc=data.get("initial_pixel_rc"),
                    confirmation_pixel_rc=data.get("confirmation_pixel_rc"),
                    mask_method=data.get("segment_mode"),
                    fit_quality=field.get("quality", {}),
                )
                if args.extra_features:
                    row["all_fields_status"] = data["status"]
                    row["features"] = {
                        k: {
                            "value": f["value"],
                            "warnings": f["warnings"],
                            "quality": f["quality"],
                        }
                        for k, f in data.get("geometry", {}).items()
                    }
                dest.write(json.dumps(row, allow_nan=False) + "\n")
                dest.flush()


if __name__ == "__main__":
    main()
