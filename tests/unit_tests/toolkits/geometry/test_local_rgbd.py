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

"""Offline regressions for local object measurements and typed local point selection."""

import numpy as np
import pytest
from pydantic import ValidationError

from rpent.robots.components.molmo_client import MolmoResult
from toolkits.geometry.local_point_agent import LocalPointAgent, PointOutput
from toolkits.geometry.rgbd_parts import measure


def test_far_color_target_needs_one_occluding_depth_contour():
    from toolkits.geometry.chromatic_region import colored_region, described_hue

    rgb = np.full((100, 150, 3), 130, dtype=np.uint8)
    depth = np.ones(rgb.shape[:2])
    valid = np.ones_like(depth, dtype=bool)
    rgb[24:48, 90:118] = [220, 12, 10]
    hue = described_hue("the red rectangular block")
    assert colored_region(rgb, depth, valid, [38, 15], hue) is None
    depth[24:48, 90:118] = 0.6
    mask, core, evidence = colored_region(rgb, depth, valid, [38, 15], hue)
    assert evidence["search_scope"] == "unique_occluding_contour" and mask[tuple(core)]
    assert described_hue("red and blue box") is None
    rgb[64:86, 80:102] = [220, 12, 10]
    depth[64:86, 80:102] = 0.65
    assert colored_region(rgb, depth, valid, [38, 15], hue) is None


def test_local_agent_preserves_missing_point_and_raster():
    class Missing:
        def ground(self, image, query):
            assert image == b"rgb" and query == "blue cup"
            return MolmoResult(False, image_size=(13, 9), answer="not found")

    got = LocalPointAgent(Missing()).ground(b"rgb", "blue cup")
    assert not got.found and got.point_xy is None and got.image_size == (13, 9)


@pytest.mark.parametrize(
    "mode", ["surface_point", "visible_region", "rectangle_center"]
)
def test_object_identity_gates_every_coordinate_return(tmp_path, mode):
    import io
    from types import SimpleNamespace

    from PIL import Image

    from rpent.robots.components.sam3_client import Sam3Result
    from rpent.session import EnvState
    from toolkits.geometry.geometry_tool import GeometryTool
    from toolkits.geometry.observation import GeometryObservation

    class Backend:
        answer = "No."
        point = (3.0, 2.0)

        def ground(self, image, query):
            return MolmoResult(True, self.point, (110, 70))

        def verify(self, image, query, *, isolated=False):
            assert Image.open(io.BytesIO(image)).size == (
                (320, 320) if isolated else (640, 320)
            )
            if query is None:
                return "a toy block"
            return self.answer

    h, w = 70, 110
    rgb = np.full((h, w, 3), 140, np.uint8)
    rr, cc = np.indices((h, w))
    world = np.stack((cc * 0.002, rr * 0.002, rr * 0.0), axis=2)
    cam = np.diag([1.0, -1.0, -1.0, 1.0])
    cam[2, 3] = 1.0

    def cube(left):
        rgb[25:44, left : left + 19] = [215, 15, 13]
        world[25:44, left : left + 19, 2] = 0.038

    cube(63)

    class Provider:
        def get_observation(self, *, camera, step):
            return GeometryObservation(
                camera,
                0,
                "bench",
                rgb,
                world_xyz=world,
                camera_to_world=cam,
                up_direction=np.array([0.0, 0.0, 1.0]),
            )

    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    backend = Backend()
    tool = GeometryTool(
        state=state,
        observations=Provider(),
        grounder=LocalPointAgent(backend),
        object_first=True,
        region_source="depth",
    )

    def request():
        args = (
            {"required_geometry": [mode]}
            if mode == "rectangle_center"
            else {"measurement": mode}
        )
        return tool.geometry("the red cube", "view", step=0, **args).data

    def check_unknown(result):
        assert result["status"] == "insufficient_evidence" and not result["found"]
        assert result["world_xyz"] is None and result.get("pixel_rc") is None
        assert (
            result.get("surface_point_xyz") is None
            and result.get("visible_bounds_xyz") is None
        )
        assert all(f["value"] is None for f in result.get("geometry", {}).values())
        assert "observation_request" not in result and "inspection_anchor" not in result
        assert [a["camera"] for a in result["view_evidence"]["attempts"]] == ["view"]

    check_unknown(request())
    backend.answer = "Yes, maybe."
    check_unknown(request())
    backend.answer = "Yes."
    positive = request()
    assert positive["found"] and 63 <= positive["pixel_rc"][1] < 82
    assert positive["calls"]["grounding"] >= 1
    if mode == "surface_point":
        backend.point = (72.0, 34.0)
        # Top-face geometry remains available when the requested rim is not visible.
        mixed = tool.geometry(
            "the red cube",
            "view",
            0,
            required_geometry=["rectangle_center", "rim_center"],
        ).data
        assert mixed["status"] == "partial", mixed
        assert mixed["geometry"]["rectangle_center"]["value"] is not None
        assert mixed["geometry"]["rim_center"]["value"] is None
        assert (
            mixed["geometry"]["rim_center"]["reason"] == "part_segmentation_unavailable"
        )
        legacy = tool.geometry(
            "the red cube", "view", 0, required_geometry=["rim_center"]
        ).data
        check_unknown(legacy)
        assert (
            legacy["geometry"]["rim_center"]["reason"]
            == "part_segmentation_unavailable"
        )

        def segment_top(image, **kwargs):
            return Sam3Result(found=True, mask=world[:, :, 2] > 0, score=0.95)

        with_sam = GeometryTool(
            state=state,
            observations=Provider(),
            grounder=LocalPointAgent(backend),
            segmenter=SimpleNamespace(segment=segment_top),
            region_source="depth",
        )
        both = with_sam.geometry(
            "the red cube",
            "view",
            required_geometry=["rectangle_center", "plane_normal"],
        ).data
        assert both["status"] == "warning" and both["calls"]["segment"] > 0
        center = both["geometry"]["rectangle_center"]
        assert not center["fit_accepted"]
        assert "region_selected_by_described_color_and_depth" in center["warnings"]
        np.testing.assert_allclose(center["value"], [0.144, 0.068, 0.038], atol=1e-6)
        assert both["geometry"]["plane_normal"]["fit_accepted"]
        np.testing.assert_allclose(
            both["geometry"]["plane_normal"]["value"], [0, 0, 1], atol=1e-6
        )
        mismatched = tool.geometry("the bar soap", "view", step=0).data
        check_unknown(mismatched)
        assert mismatched["reason"] == "independent_object_name_not_confirmed"
        wrong_class = tool.geometry("the small white fluted bowl", "view", step=0).data
        check_unknown(wrong_class)
        assert wrong_class["reason"] == "independent_object_name_not_confirmed"
        assert tool.geometry("the toy block", "view", step=0).data["found"]
    cube(15)
    duplicate = request()
    check_unknown(duplicate)
    assert duplicate["reason"] == "multiple_matching_colored_boxes"
    if mode == "surface_point":
        backend.point = (72.0, 34.0)
        same_appearance = tool.geometry(
            "the plain toy on the table", "view", step=0
        ).data
        check_unknown(same_appearance)
        assert same_appearance["reason"] == "another_visually_compatible_instance"
        # A nearer neutral panel hides half of the second matching surface.
        rgb[23:46, 25:37] = [125, 125, 125]
        world[23:46, 25:37, 2] = 0.091
        covered = tool.geometry("the plain toy on the table", "view", step=0).data
        check_unknown(covered)
        assert covered["reason"] == "another_occluded_compatible_instance"


def test_full_cuboid_requires_agreement_between_top_and_observed_support():
    from toolkits.geometry.rgbd_parts import cuboid_vertices

    top = np.array([[0, 0, 0.8], [0.04, 0, 0.8], [0, 0.05, 0.8], [0.04, 0.05, 0.8]])
    up = np.array([0.0, 0.0, 1.0])
    got = np.asarray(cuboid_vertices(top, up, 0.74, 0.06))
    assert np.allclose(got[:4], top) and np.allclose(got[4:], top - [0, 0, 0.06])
    assert cuboid_vertices(top, up, None, 0.06) is None
    assert cuboid_vertices(top, up, 0.5, 0.06) is None


def test_occluding_edge_needs_a_flat_side_on_the_measured_material():
    from toolkits.geometry.chromatic_region import _has_flat_side

    theta, z = np.meshgrid(np.linspace(-1.2, 1.2, 61), np.linspace(0.0, 0.04, 31))
    mask = np.ones_like(theta, dtype=bool)
    up = np.array([0.0, 0.0, 1.0])
    flat = np.stack((0.02 * theta, np.zeros_like(z), z), axis=-1)
    curved = np.stack((0.02 * np.sin(theta), 0.02 * np.cos(theta), z), axis=-1)
    assert _has_flat_side(flat, mask, up)
    assert not _has_flat_side(curved, mask, up)


def test_top_edges_stay_connected_to_the_observed_top():
    from toolkits.geometry.chromatic_region import _top_footprint

    y, x = np.indices((45, 40))
    world = np.stack((x * 0.001, y * 0.001, np.zeros_like(x)), axis=-1)
    mask = np.zeros(x.shape, dtype=bool)
    mask[8:30, 6:22] = True
    world[8:24, 6:22, 2] = 0.04
    valid = np.ones_like(mask)
    up = np.array([0.0, 0.0, 1.0])
    footprint = _top_footprint(world, valid, mask, up)
    assert footprint[0] > 0.95 and footprint[1] > 0.95 and footprint[2] == 2

    # A second top is beside the bottom of the first box's projected wall.
    world[31:36, 6:13, 2] = 0.04
    assert _top_footprint(world, valid, mask, up) == footprint


def test_separate_color_sliver_survives_the_ambiguity_check():
    from toolkits.geometry.chromatic_region import _unresolved_fragment

    labels = np.zeros((50, 65), dtype=int)
    labels[10:30, 10:30] = 1
    material = labels.astype(bool)
    depth = np.full(labels.shape, 0.9)
    valid = np.ones_like(material)
    depth[18:35, 50:] = 0.76
    assert not _unresolved_fragment(material, labels, 1, depth, valid)
    material[20:30, 48:50] = True
    assert _unresolved_fragment(material, labels, 1, depth, valid)


def test_oblique_can_uses_upper_lid_when_more_wall_is_visible():
    rng = np.random.default_rng(0)
    angle = rng.uniform(-0.9, 0.9, 750)
    height = rng.uniform(0.018, 0.075, len(angle))
    wall = np.c_[0.032 * np.cos(angle), 0.032 * np.sin(angle), height]
    angle = np.linspace(-1.5, 1.5, 24)
    rim = np.c_[
        0.032 * np.cos(angle),
        0.032 * np.sin(angle),
        0.075 + rng.normal(0, 0.00045, len(angle)),
    ]
    cloud = np.vstack((wall, rim))
    cloud += rng.normal(0, 0.0006, cloud.shape)
    center, quality = measure(cloud, [0, 0, 1], "lid_center")
    assert np.linalg.norm(np.asarray(center) - [0, 0, 0.075]) < 0.005
    assert "upper_lid_plane_selected_above_side_wall" in quality["warnings"]


def test_visible_cup_lip_ignores_floor_and_lower_handle():
    rng = np.random.default_rng(44)
    angle = rng.uniform(0, 2 * np.pi, 6500)
    z = rng.uniform(0.002, 0.076, len(angle))
    r = 0.041
    walls = np.c_[r * np.cos(angle), r * np.sin(angle), z]
    a = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    lip = np.c_[r * np.cos(a), r * np.sin(a), np.full(len(a), 0.076)]
    handle = np.c_[
        rng.uniform(0.04, 0.058, 500),
        rng.uniform(-0.003, 0.003, 500),
        rng.uniform(0.02, 0.055, 500),
    ]
    pts = np.vstack((walls, lip, handle)) + [0.14, -0.22, 0.83]
    value, _ = measure(
        pts, [0, 0, 1], "rim_extrema", np.array([1.0, 0, 0]), np.array([0.0, -1, 0])
    )
    expected = np.array(
        [
            [0.14 - r, -0.22, 0.906],
            [0.14 + r, -0.22, 0.906],
            [0.14, -0.22 - r, 0.906],
            [0.14, -0.22 + r, 0.906],
        ]
    )
    assert np.max(np.linalg.norm(np.array(value) - expected, axis=1)) < 0.003


def test_rectangular_face_size_and_corners_follow_camera_frame():
    x, y = np.meshgrid(np.linspace(-0.049, 0.049, 160), np.linspace(-0.024, 0.024, 110))
    face = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.31)]
    value, _ = measure(
        face, [0, 0, 1], "top_corners", np.array([1.0, 0, 0]), np.array([0.0, -1, 0])
    )
    expected = np.array(
        [
            [-0.049, 0.024, 0.31],
            [0.049, 0.024, 0.31],
            [-0.049, -0.024, 0.31],
            [0.049, -0.024, 0.31],
        ]
    )
    assert np.max(np.linalg.norm(np.array(value) - expected, axis=1)) < 0.002
    size, _ = measure(face, [0, 0, 1], "top_size")
    np.testing.assert_allclose(size, [0.098, 0.048], atol=0.002)


def test_native_point_schema_rejects_nonfinite_pixels_and_empty_images():
    with pytest.raises(ValidationError):
        PointOutput(xy=(float("nan"), 1), image_size=(4, 4))
    with pytest.raises(ValidationError):
        PointOutput(xy=None, image_size=(4, 0))


def test_self_selected_crop_projects_back_at_image_boundary():
    import io

    from PIL import Image

    from toolkits.geometry.refined_grounder import RefinedGrounder

    buf = io.BytesIO()
    Image.new("RGB", (120, 80)).save(buf, "PNG")

    class Backend:
        def ground(self, image, query):
            assert Image.open(io.BytesIO(image)).size == (60, 40)
            assert query == "cup"
            return MolmoResult(True, (55.5, 20.0), (60, 40), "cup")

    result, trace = RefinedGrounder(Backend()).refine(
        buf.getvalue(), "cup", MolmoResult(True, (115.0, 4.0), (120, 80), "cup")
    )
    assert trace["crop"] == (60, 0, 120, 40)
    assert result.point_xy == (115.5, 20.0) and result.image_size == (120, 80)


def test_observation_accepts_native_float32_camera_but_rejects_shear():
    from toolkits.geometry.observation import GeometryObservation

    cam = np.eye(4)
    cam[:3, :3] = [
        [0.891578717, 0.223093904, -0.394101874],
        [0.452865522, -0.439216035, 0.775887842],
        [-5.447198e-8, -0.870240425, -0.492627258],
    ]
    inputs = {
        "camera": "head",
        "step": 0,
        "world_frame_id": "scene",
        "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        "depth_m": np.ones((2, 2)),
        "intrinsics": np.eye(3),
        "camera_to_world": cam,
    }
    assert GeometryObservation(**inputs).valid_mask.all()
    cam[0, 1] += 0.002
    with pytest.raises(ValueError, match="rigid"):
        GeometryObservation(**inputs)


def test_known_dimensions_recover_clipped_top_and_reject_wrong_face():
    from toolkits.geometry.registration import constrained_rectangle

    x, y = np.meshgrid(np.linspace(-0.05, 0.015, 88), np.linspace(-0.02, 0.02, 66))
    face = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.35)]
    camera = np.array([-0.4, -0.2, 0.7])
    axes = (np.array([1, 0, 0]), np.array([0, -1, 0]))
    result, info = constrained_rectangle(
        (0.10, 0.04), face, [0, 0, 1], camera, camera_basis=axes
    )
    expected = [
        [-0.05, 0.02, 0.35],
        [0.05, 0.02, 0.35],
        [-0.05, -0.02, 0.35],
        [0.05, -0.02, 0.35],
    ]
    np.testing.assert_allclose(result, expected, atol=0.002)
    assert "known_rectangular_face_dimensions" in info["warnings"]
    missing, diagnostic = constrained_rectangle((0.03, 0.02), face, [0, 0, 1], camera)
    assert missing is None and diagnostic["reason"] == "inconsistent_face_size"


@pytest.mark.parametrize("duplicate", [False, True])
def test_rectangle_identity_requires_exactly_one_measured_match(duplicate):
    from types import SimpleNamespace

    from toolkits.geometry.rectangle_evidence import reconcile

    y, x = np.indices((180, 250))
    world = np.stack((x * 0.001, y * 0.001, np.zeros(x.shape)), axis=-1)
    anchor_mask = (x >= 20) & (x < 43) & (y >= 20) & (y < 37)
    world[anchor_mask, 2] = 0.035
    world[55:95, 65:129, 2] = 0.025
    if duplicate:
        world[105:145, 150:214, 2] = 0.025
    cam = np.diag([1.0, -1, -1, 1])
    cam[2, 3] = 0.8
    obs = SimpleNamespace(
        camera_to_world=cam,
        up_direction=np.array([0, 0, 1]),
        world_xyz=world,
        valid_mask=np.ones(x.shape, dtype=bool),
    )
    clouds, _, info = reconcile(
        obs,
        {"depth": world[anchor_mask]},
        {"depth": anchor_mask},
        (0.064, 0.040),
        0.0,
        [25, 30],
    )
    if duplicate:
        assert not clouds and info["reason"] == "known_dimensions_not_uniquely_observed"
    else:
        assert len(clouds["depth"]) > 100 and info["original_anchor_distance_px"] > 25


def test_explicit_relative_lightness_distinguishes_adjacent_components():
    from toolkits.geometry.depth_foreground import described_lightness, foreground

    y, x = np.indices((160, 180))
    world = np.stack((x * 0.001, y * 0.001, np.zeros(x.shape)), axis=-1)
    world[65:90, 65:85, 2] = 0.025
    world[65:90, 87:107, 2] = 0.025
    rgb = np.full((*x.shape, 3), 80, dtype=np.uint8)
    rgb[65:90, 87:107] = 170
    cam = np.diag([1.0, -1, -1, 1])
    cam[2, 3] = 1
    pref = described_lightness("the light-colored object")
    assert pref == 1 and described_lightness("the black-and-white pot") == 0
    mask, info = foreground(
        world,
        np.ones(x.shape, dtype=bool),
        np.array([0, 0, 1]),
        cam,
        [77, 83],
        rgb=rgb,
        lightness=pref,
    )
    assert mask[77, 95] and not mask[77, 74] and info["selected_by_relative_lightness"]


def test_two_cartons_sharing_a_depth_component_have_separate_label_regions():
    from toolkits.geometry.observation import GeometryObservation
    from toolkits.geometry.package_identity import _candidate_faces

    yy, xx = np.indices((120, 160))
    left = (xx >= 18) & (xx < 59) & (yy >= 30) & (yy < 65)
    right = (xx >= 72) & (xx < 118) & (yy >= 34) & (yy < 62)
    bridge = (xx >= 58) & (xx <= 72) & (yy >= 47) & (yy < 50)
    height = np.where(left | right, 0.025, np.where(bridge, 0.01, 0.0))
    world = np.stack((xx * 0.001, yy * 0.001, height), axis=2)
    cam = np.diag([1.0, -1.0, -1.0, 1.0])
    cam[2, 3] = 1.0
    obs = GeometryObservation(
        "front",
        0,
        "table",
        np.zeros((*xx.shape, 3), np.uint8),
        world_xyz=world,
        camera_to_world=cam,
        up_direction=np.array([0.0, 0.0, 1.0]),
    )
    components = np.asarray(left | right | bridge, dtype=int)
    candidates = list(
        _candidate_faces(obs, components, {"blue": xx % 2 == 0, "white": xx % 2 == 1})
    )
    complete = [r for r, _, _, face, _ in candidates if face is not None]
    assert len(complete) == 2
    assert all(not (np.any(mask & left) and np.any(mask & right)) for mask in complete)


def test_floating_reference_distinguishes_height_from_distance_above_support():
    from toolkits.geometry.evaluation.feature_metrics import reference_values

    profile = {"local_center": [0, 0, 0], "hull": [[0, 0, 0], [0, 0, 0.1]]}
    pose = np.eye(4)
    pose[2, 3] = 0.5
    ref = reference_values(
        profile, pose, np.eye(4), np.array([0, 0, 1]), support_height_m=0.4
    )
    assert ref["object_height"] == pytest.approx(0.1)
    assert ref["height_above_support"] == pytest.approx(0.2)
    assert "height_above_support" not in reference_values(
        profile, pose, np.eye(4), np.array([0, 0, 1])
    )


@pytest.mark.parametrize("second_open", [False, True])
def test_plain_cup_checks_the_second_cavity_even_if_the_visual_model_picks_one(
    second_open,
):
    from toolkits.geometry.observation import GeometryObservation
    from toolkits.geometry.target_identity import _other_plain_vessel

    yy, xx = np.indices((100, 180))
    radii = [np.hypot(xx - x, yy - 50) for x in (45, 131)]
    depth = np.ones_like(yy, dtype=float)
    for i, rad in enumerate(radii):
        depth[rad <= 23] = 0.90
        if i == 0 or second_open:
            depth[rad < 15] = 0.95
    cam = np.diag([1.0, -1.0, -1.0, 1.0])
    cam[2, 3] = 1
    obs = GeometryObservation(
        "view",
        0,
        "desk",
        np.full((*yy.shape, 3), 140, np.uint8),
        depth_m=depth,
        intrinsics=np.array([[400, 0, 90], [0, 400, 50], [0, 0, 1]]),
        camera_to_world=cam,
        up_direction=np.array([0.0, 0.0, 1.0]),
    )
    assert (
        _other_plain_vessel(
            obs, radii[0] <= 23, 0.0, "the pale cup with an open round rim"
        )
        is second_open
    )


@pytest.mark.parametrize(
    ("name", "peer", "accepted"),
    [("bowl", False, True), ("ramekin", False, False), ("bowl", True, False)],
)
def test_between_checks_the_target_name_and_the_landmark_relation(
    monkeypatch, name, peer, accepted
):
    from types import SimpleNamespace

    from toolkits.geometry import target_identity

    query = "patterned black bowl between the red-rim plate and the silver gray ramekin"
    mask = np.zeros((100, 120), bool)
    mask[30:70, 10:50] = True
    other = np.roll(mask, 60, axis=1)
    obs = SimpleNamespace(
        valid_mask=np.ones_like(mask), rgb=np.full((100, 120, 3), 128, np.uint8)
    )
    monkeypatch.setattr(
        target_identity, "_opening_evidence", lambda *args: {"resolved": True}
    )
    monkeypatch.setattr(
        target_identity,
        "similar_instances",
        lambda *args: [(other, False)] if peer else [],
    )
    questions = []

    def verify(image, prompt, *, isolated=False):
        questions.append((prompt, isolated))
        return name if prompt is None else "yes"

    out = target_identity.verify_region(
        obs, mask, query, SimpleNamespace(verify=verify), {"calls": {}}, [45, 20]
    )
    assert out["accepted"] is accepted
    assert query in questions[0][0] and questions[0][1] is False
    assert (None, True) in questions
    if name == "bowl":
        assert ("patterned black bowl", True) in questions
    if name == "ramekin":
        assert out["reason"] == "independent_object_name_not_confirmed"
    if peer:
        assert (
            out["reason"] == "another_visually_compatible_instance"
            and query in questions[-1][0]
        )
