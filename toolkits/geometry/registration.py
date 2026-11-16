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

"""Register an optional, caller-supplied metric surface model to an observed object cloud."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from toolkits.geometry.rgbd_parts import measure


def _visible(points, towards, pitch):
    towards = towards / np.linalg.norm(towards)
    axis = np.eye(3)[np.argmin(abs(towards))]
    right = np.cross(axis, towards)
    right /= np.linalg.norm(right)
    xy = points @ np.stack((right, np.cross(towards, right)), axis=1)
    ij = np.floor(xy / pitch).astype(int)
    ij -= ij.min(axis=0)
    section = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    z = points @ towards
    maximum = np.full(section.max() + 1, -np.inf)
    np.maximum.at(maximum, section, z)
    return points[z >= maximum[section] - pitch * 0.8]


@dataclass
class SurfaceTemplate:
    """Static metric mesh sampled in an upright object frame, with no instance pose."""

    points: np.ndarray
    _landmarks: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        if (
            self.points.ndim != 2
            or self.points.shape[1] != 3
            or len(self.points) < 30
            or not np.isfinite(self.points).all()
        ):
            raise ValueError("template requires a finite metric surface")
        self.origin = np.mean(np.quantile(self.points, [0.001, 0.999], axis=0), axis=0)
        self.points = self.points - self.origin
        self.tree = cKDTree(self.points)

    def landmark(self, name):
        if name not in self._landmarks:
            up = np.array([0.0, 0.0, 1.0])
            pts = self.points
            if name in ("rectangle_center", "lid_center", "top_corners", "top_size"):
                ztop = np.quantile(pts[:, 2], 0.997)
                # The complete supplied template resolves edges clipped in the observation.
                high = pts[pts[:, 2] >= ztop - 0.003]
                if len(high) >= 40:
                    pts = high
            self._landmarks[name] = measure(
                pts, up, name, np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
            )
        return self._landmarks[name]


def register(template, observed, up, camera_origin):
    """Trim imperfect segmentation and match only visible model surfaces to RGB-D."""
    obs = np.asarray(observed, dtype=float)
    if len(obs) < 28 or camera_origin is None or up is None:
        return None
    rng = np.random.default_rng(81)
    sample = obs[rng.choice(len(obs), min(1300, len(obs)), replace=False)]
    mid = np.median(sample, axis=0)
    extent = np.linalg.norm(np.ptp(template.points, axis=0))
    d = np.linalg.norm(sample - mid, axis=1)
    sample = sample[d < extent * 1.25]
    if len(sample) < 25:
        return None
    cam = np.asarray(camera_origin)
    unit_up = np.asarray(up, dtype=float)
    unit_up = unit_up / np.linalg.norm(unit_up)
    basis = Rotation.align_vectors([unit_up], [[0.0, 0.0, 1.0]])[0].as_matrix()
    view = cam - mid
    pitch = max(0.0013, extent / 85)
    candidates = []
    model = template.points[::2]
    scalar = np.quantile(sample, [0.02, 0.98], axis=0).mean(axis=0)
    for yaw in np.arange(0, 360, 30):
        R = basis @ Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        visible = _visible(model, view @ R, pitch)
        v = visible @ R.T
        t = scalar - np.quantile(v, [0.02, 0.98], axis=0).mean(axis=0)
        for stage in range(2):
            trial = sample[::3] if stage == 0 else sample
            for _ in range(11 if stage == 0 else 15):
                local = (trial - t) @ R
                ds, idx = template.tree.query(local)
                bound = max(0.0017, np.quantile(ds, 0.83))
                keep = ds <= bound
                a = template.points[idx[keep]]
                b = trial[keep]
                if len(a) < 10:
                    break
                aa, bb = a.mean(0), b.mean(0)
                u, _, vt = np.linalg.svd((a - aa).T @ (b - bb))
                fix = np.diag([1, 1, np.linalg.det(vt.T @ u.T)])
                new = vt.T @ fix @ u.T
                if new[:, 2] @ unit_up < 0.83:
                    break
                t = bb - new @ aa
                R = new
        ds, _ = template.tree.query((sample - t) @ R)
        # The reverse distance is computed for visible model surface samples only.
        vis = _visible(model, (cam - t) @ R, pitch)[::6] @ R.T + t
        rev = cKDTree(sample).query(vis)[0]
        score = (
            np.quantile(ds, 0.65)
            + 0.48 * np.quantile(rev, 0.65)
            + 0.12 * np.quantile(ds, 0.90)
        )
        candidates.append(
            (score, R, t, float(np.quantile(ds, 0.85)), float(np.mean(ds < 0.004)))
        )
    best = min(candidates, key=lambda c: c[0])
    if best[4] < 0.30:
        return None
    return (
        best[1],
        best[2],
        {
            "registration_score_m": float(best[0]),
            "p85_surface_residual_m": best[3],
            "observed_fraction_within_4mm": best[4],
            "n_samples": len(sample),
        },
    )


def registered_center(template, observed, up, camera_origin, quantity):
    result = register(template, observed, up, camera_origin)
    if result is None:
        return None, {"reason": "insufficient_model_to_scene_agreement"}
    R, t, q = result
    local, metrics = template.landmark(quantity)
    if local is None:
        return None, {"reason": "static_mesh_feature_not_resolved"}
    center = (R @ np.asarray(local) + t).tolist()
    q["warnings"] = ["known_metric_object_template"]
    return center, q


def constrained_rectangle(template, points, up, camera_origin, *, camera_basis=None):
    """Reconstruct a clipped extent using a physically supported edge and known dimensions."""
    from scipy.spatial import ConvexHull, QhullError

    from toolkits.geometry.geometry import _basis
    from toolkits.geometry.rgbd_parts import _face

    face = _face(points, up)
    if face is None:
        return None, {"reason": "missing_top_face"}
    p, n, surf = face
    ax = np.stack(_basis(n), axis=1)
    flat = (surf - p) @ ax
    try:
        verts = flat[ConvexHull(flat).vertices]
    except (ValueError, QhullError):
        return None, {"reason": "insufficient_top_face"}
    edges = np.diff(verts, axis=0, append=verts[:1])
    angles = np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2)
    angles = np.unique(np.round(angles, 3))
    # Fit directions to long straight portions of the visible rectangle.
    best = None
    for angle in angles:
        ds = ax @ np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        xy = (surf - p) @ ds
        lo, hi = np.quantile(xy, [0.001, 0.999], axis=0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, ds, lo, hi)
    _, ds, lo, hi = best
    known = (
        template.landmark("top_size")[0]
        if isinstance(template, SurfaceTemplate)
        else template
    )
    if known is None:
        return None, {"reason": "known_dimensions_unavailable"}
    known = np.asarray(known, dtype=float)
    if known.shape != (2,) or not np.isfinite(known).all() or (known <= 0).any():
        raise ValueError(
            "known rectangular face dimensions must be two positive metre lengths"
        )
    known = np.sort(known)[::-1]
    full = np.empty(2)
    full[np.argsort(hi - lo)[::-1]] = known
    if np.any((hi - lo) > full + 0.012) or np.any(hi - lo < full * 0.32):
        return None, {"reason": "inconsistent_face_size"}
    xy = (points - p) @ ds
    lower = (points - p) @ n < -0.004
    center = (hi + lo) / 2
    for j in range(2):
        if full[j] - (hi[j] - lo[j]) < 0.003:
            continue
        near_lo = np.count_nonzero(lower & (abs(xy[:, j] - lo[j]) < 0.003))
        near_hi = np.count_nonzero(lower & (abs(xy[:, j] - hi[j]) < 0.003))
        if max(near_hi, near_lo) > max(12, 2 * min(near_hi, near_lo)):
            high = near_hi > near_lo
        else:
            # A face facing the camera is visible when the opposite edge is clipped.
            high = (camera_origin - p) @ ds[:, j] > 0
        center[j] = hi[j] - full[j] / 2 if high else lo[j] + full[j] / 2
    value = p + ds @ center
    if camera_basis is not None:
        right, toward = camera_basis
        signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        corners = value + (signs * full / 2) @ ds.T
        order = np.argsort((corners - value) @ toward)
        far, near = corners[order[:2]], corners[order[2:]]
        value = np.vstack(
            (far[np.argsort(far @ right)], near[np.argsort(near @ right)])
        )
    return value.tolist(), {
        "warnings": ["known_rectangular_face_dimensions"],
        "observed_extents_m": (hi - lo).tolist(),
        "known_extents_m": full.tolist(),
    }
