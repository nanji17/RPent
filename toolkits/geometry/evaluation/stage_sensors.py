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

"""Run local visual models sequentially on a single GPU. Store only model evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("molmo", "sam"))
    parser.add_argument(
        "items", type=Path, help="JSON list of image paths, IDs and object descriptions"
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("checkpoint")
    parser.add_argument("--backend", choices=("cuda", "mlx"), default="cuda")
    parser.add_argument(
        "--refine-from",
        type=Path,
        help="Reuse initial model points from the same images and prompts, then point again in a self-selected crop",
    )
    args = parser.parse_args()
    if args.stage == "sam" and args.backend != "cuda":
        parser.error("the SAM3 stage uses its CUDA runtime")
    if args.refine_from and (
        args.stage != "molmo" or args.database.resolve() == args.refine_from.resolve()
    ):
        parser.error("refinement requires a separate Molmo output database")
    items = json.loads(args.items.read_text())
    db = sqlite3.connect(args.database, timeout=60)
    db.execute(
        "CREATE TABLE IF NOT EXISTS visual (id TEXT PRIMARY KEY, rgb_sha TEXT, query TEXT, xy TEXT, answer TEXT, mask BLOB, score REAL, method TEXT)"
    )
    if "model_query" not in {col[1] for col in db.execute("PRAGMA table_info(visual)")}:
        db.execute("ALTER TABLE visual ADD COLUMN model_query TEXT")
    if "trace" not in {col[1] for col in db.execute("PRAGMA table_info(visual)")}:
        db.execute("ALTER TABLE visual ADD COLUMN trace TEXT")
    db.execute(
        "CREATE TABLE IF NOT EXISTS configuration (stage TEXT PRIMARY KEY, checkpoint TEXT)"
    )
    checkpoint = str(Path(args.checkpoint).resolve())
    previous = db.execute(
        "SELECT checkpoint FROM configuration WHERE stage=?", (args.stage,)
    ).fetchone()
    if previous and previous[0] != checkpoint:
        raise ValueError(
            "the database already contains another checkpoint for this stage"
        )
    db.execute(
        "INSERT OR IGNORE INTO configuration VALUES (?,?)", (args.stage, checkpoint)
    )
    db.commit()
    initial = None
    if args.refine_from:
        initial = sqlite3.connect(
            f"file:{args.refine_from.resolve()}?mode=ro", uri=True
        )
        first_model = initial.execute(
            "SELECT checkpoint FROM configuration WHERE stage='molmo'"
        ).fetchone()
        if first_model != (checkpoint,):
            raise ValueError(
                "initial evidence and refinement must use the same checkpoint"
            )
    if args.stage == "molmo":
        variant = "self_crop_0.50" if initial else "full_image"
        previous_variant = db.execute(
            "SELECT checkpoint FROM configuration WHERE stage='point_strategy'"
        ).fetchone()
        if previous_variant and previous_variant != (variant,):
            raise ValueError("the database already uses a different point strategy")
        db.execute(
            "INSERT OR IGNORE INTO configuration VALUES ('point_strategy',?)",
            (variant,),
        )
        db.commit()
    if args.stage == "molmo" and args.backend == "mlx":
        from toolkits.geometry.evaluation.mlx_molmo import MlxMolmoGrounder

        point = MlxMolmoGrounder(args.checkpoint).ground
    elif args.stage == "molmo":
        from rpent.robots.components.molmo_client import MolmoResult
        from rpent.robots.components.molmo_server import MolmoFacade

        model = MolmoFacade(args.checkpoint)

        def point(image, query):
            response = model._ground_bytes(image, query, point_mode="native")
            return MolmoResult(
                response.point_xy is not None,
                tuple(response.point_xy) if response.point_xy is not None else None,
                tuple(response.image_size),
                response.answer,
            )
    else:
        from rpent.robots.components.sam3_server import Sam3Facade

        model = Sam3Facade(args.checkpoint)
    if initial is not None:
        from types import SimpleNamespace

        from rpent.robots.components.molmo_client import MolmoResult as Pixel
        from toolkits.geometry.refined_grounder import RefinedGrounder

        refiner = RefinedGrounder(SimpleNamespace(ground=point))
    for n, item in enumerate(items, 1):
        key, query = item["id"], item["query"]
        model_query = item.get("model_query", query)
        image = Path(item["image"]).read_bytes()
        digest = hashlib.sha256(image).hexdigest()
        row = db.execute(
            "SELECT rgb_sha, query, xy, method, COALESCE(model_query,query) FROM visual WHERE id=?",
            (key,),
        ).fetchone()
        if row and (row[:2] != (digest, query) or row[4] != model_query):
            raise ValueError(f"cached evidence does not match {key}")
        if args.stage == "molmo":
            if row:
                continue
            same = db.execute(
                "SELECT xy,answer,trace FROM visual WHERE rgb_sha=? AND COALESCE(model_query,query)=? LIMIT 1",
                (digest, model_query),
            ).fetchone()
            if same:
                xy, answer, trace = same
            else:
                trace = None
                if initial is not None:
                    prior = initial.execute(
                        "SELECT rgb_sha,COALESCE(model_query,query),xy,answer FROM visual WHERE id=?",
                        (key,),
                    ).fetchone()
                    if prior is None or prior[:2] != (digest, model_query):
                        raise ValueError(f"initial evidence does not match {key}")
                    first_xy = json.loads(prior[2])
                    size = Image.open(io.BytesIO(image)).size
                    previous_point = Pixel(
                        first_xy is not None,
                        tuple(first_xy) if first_xy is not None else None,
                        size,
                        prior[3],
                    )
                    ans, provenance = refiner.refine(image, model_query, previous_point)
                    trace = json.dumps(provenance)
                else:
                    ans = point(image, model_query)
                xy, answer = json.dumps(ans.point_xy), ans.answer
            db.execute(
                "INSERT INTO visual (id,rgb_sha,query,xy,answer,model_query,trace) VALUES (?,?,?,?,?,?,?)",
                (key, digest, query, xy, answer, model_query, trace),
            )
        else:
            if not row:
                raise ValueError(f"missing Molmo stage for {key}")
            if row[3] is not None:
                continue
            import base64

            xy = json.loads(row[2])
            mask, score, method = None, None, "no_point"
            if xy is not None:
                rc = [round(xy[1]), round(xy[0])]
                payload = base64.b64encode(image).decode("ascii")
                for style, params in (
                    ("text", {"text_prompt": query}),
                    ("point", {"point": rc}),
                ):
                    res = model.segment(payload, **params)
                    if res.get("found"):
                        data = base64.b64decode(res["mask_png_base64"])
                        with Image.open(io.BytesIO(data)) as decoded:
                            if decoded.getpixel((rc[1], rc[0])):
                                mask, score, method = data, res["score"], style
                                break
                    method = "no_consistent_mask"
            db.execute(
                "UPDATE visual SET mask=?,score=?,method=? WHERE id=?",
                (mask, score, method, key),
            )
        db.commit()
        if n % 5 == 0 or n == len(items):
            print(f"{args.stage} {n}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
