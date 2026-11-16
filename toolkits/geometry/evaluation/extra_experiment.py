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

"""Measure supplementary RGB-D frames through the same registered geometry tool."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from rpent.session import EnvState
from toolkits.geometry.evaluation.local_experiment import RecordedObject
from toolkits.geometry.evaluation.replay import ReplayGrounder
from toolkits.geometry.geometry_tool import GeometryTool
from toolkits.geometry.local_point_agent import LocalPointAgent
from toolkits.geometry.observation import GeometryObservation


class Pinned:
    def __init__(self, obs):
        self.obs = obs

    def get_observation(self, *, camera, step):
        if camera != self.obs.camera or step not in (-1, self.obs.step):
            raise ValueError("different sensor frame requested")
        return self.obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sensors",
        type=Path,
        help="List of frames, query, camera, step and requested fields only",
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--identity-replay",
        type=Path,
        help="Directory of recorded candidate-verification calls",
    )
    parser.add_argument(
        "--region", choices=("sam", "depth", "hybrid"), default="hybrid"
    )
    args = parser.parse_args()
    identities = ReplayGrounder(args.identity_replay) if args.identity_replay else None
    requests = json.loads(args.sensors.read_text())
    db = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    with (
        TemporaryDirectory(prefix="geometry-extra-") as root,
        args.predictions.open("x") as stream,
    ):
        state = EnvState(Path(root))
        for _ in range(1 + max(r["step"] for r in requests)):
            with state.record_step(state={}):
                pass
        for r in requests:
            if r.keys() - {"id", "query", "frame", "camera", "step", "fields"}:
                raise ValueError("unknown keys in sensor-only experiment request")
            frame = Path(r["frame"])
            rgb = Path(str(frame) + ".png").read_bytes()
            with np.load(str(frame) + ".npz", allow_pickle=False) as z:
                obs = GeometryObservation(
                    camera=r["camera"],
                    step=r["step"],
                    world_frame_id=frame.name,
                    rgb=np.asarray(Image.open(str(frame) + ".png").convert("RGB")),
                    depth_m=z["depth"],
                    intrinsics=z["K"],
                    camera_to_world=z["cam"],
                    up_direction=np.array([0.0, 0.0, 1.0]),
                )
            evidence = RecordedObject(
                rgb,
                r["query"],
                db.execute("SELECT * FROM visual WHERE id=?", (r["id"],)).fetchone(),
                identities=identities,
            )
            tool = GeometryTool(
                state=state,
                observations=Pinned(obs),
                grounder=LocalPointAgent(evidence),
                segmenter=evidence,
                object_first=True,
                region_source=args.region,
            )
            args_model = tool.geometry.args_schema.model_validate(
                {
                    "query": r["query"],
                    "camera": r["camera"],
                    "step": r["step"],
                    "required_geometry": r["fields"],
                },
                strict=True,
            )
            d = tool.geometry(**args_model.model_dump()).data
            row = {
                "id": r["id"],
                "visual_prompt": evidence.prompt,
                "step": r["step"],
                "pixel": d.get("pixel_rc"),
                "status": d["status"],
                "fields": {
                    k: {
                        "value": v["value"],
                        "warnings": v["warnings"],
                        "reason": v["reason"],
                    }
                    for k, v in d.get("geometry", {}).items()
                },
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
