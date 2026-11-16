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

"""Build an optional geometry catalog from standalone visual asset XML, without scene state."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def load_rectangles(path, manifest):
    """Bind optional standalone rectangular assets to the checklist descriptions."""
    from toolkits.geometry.registration import SurfaceTemplate

    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as models:
        return {
            manifest["references"][key]["query"]: SurfaceTemplate(models[key])
            for key in models.files
            if manifest["references"][key]["quantity"] == "rectangle_center"
        }


def read_visual(path):
    if path.suffix == ".msh":
        with path.open("rb") as f:
            nv, nn, nt, nf = np.fromfile(f, "<i4", 4)
            verts = np.fromfile(f, "<f4", 3 * nv).reshape(-1, 3)
            f.seek((3 * nn + 2 * nt) * 4, 1)
            faces = np.fromfile(f, "<i4", 3 * nf).reshape(-1, 3)
        return verts[faces]
    if path.suffix == ".stl":
        with path.open("rb") as f:
            f.seek(80)
            count = np.fromfile(f, "<u4", 1)[0]
            records = np.fromfile(
                f,
                dtype=[
                    ("normal", "<f4", 3),
                    ("xyz", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ],
                count=count,
            )
        return records["xyz"]
    raise ValueError(path.suffix)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("asset_root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    # Native visual mesh names and author-specified orientation of the sealed can.
    specs = {
        "plate": ("stable_scanned_objects/plate/plate", "model_vis"),
        "ramekin": (
            "stable_scanned_objects/glazed_rim_porcelain_ramekin/glazed_rim_porcelain_ramekin",
            "glazed_rim_porcelain_ramekin_vis",
        ),
        "black_bowl": (
            "stable_scanned_objects/akita_black_bowl/akita_black_bowl",
            "akita_black_bowl_vis",
        ),
        "cheese": ("stable_hope_objects/cream_cheese/cream_cheese", "cream_cheese_vis"),
        "soup": ("stable_hope_objects/alphabet_soup/alphabet_soup", "textured_vis"),
        "wine_bottle": (
            "turbosquid_objects/wine_bottle/wine_bottle",
            "wine_bottle_red_main_vis",
        ),
        "gray_white_mug": (
            "turbosquid_objects/porcelain_mug/porcelain_mug",
            "porcelain_mug_vis",
        ),
        "yellow_white_mug": (
            "turbosquid_objects/white_yellow_mug/white_yellow_mug",
            "white_yellow_mug_vis",
        ),
        "microwave": ("articulated_objects/microwave", "micro"),
    }
    rng = np.random.default_rng(2026)
    data = {}
    for key, (filename, mesh_name) in specs.items():
        xml = a.asset_root / (filename + ".xml")
        root = ET.parse(xml).getroot()
        mesh = next(
            m for m in root.findall(".//asset/mesh") if m.get("name") == mesh_name
        )
        path = xml.parent / mesh.get("file")
        if not path.exists():
            matches = list(a.asset_root.glob("**/" + path.name))
            if len(matches) != 1:
                raise ValueError((path, matches))
            path = matches[0]
        tri = read_visual(path) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(cross, axis=1)
        ids = rng.choice(len(tri), size=36000, p=area / area.sum())
        triangles = tri[ids]
        u, v = rng.random((2, len(ids)))
        u = np.sqrt(u)
        pts = (
            (1 - u[:, None]) * triangles[:, 0]
            + (u * (1 - v))[:, None] * triangles[:, 1]
            + (u * v)[:, None] * triangles[:, 2]
        )
        if key == "soup":
            pts = pts[:, [0, 2, 1]] * np.array([1, -1, 1])
        data[key] = pts.astype("f4")
    with a.output.open("xb") as f:
        np.savez_compressed(f, **data)


if __name__ == "__main__":
    main()
