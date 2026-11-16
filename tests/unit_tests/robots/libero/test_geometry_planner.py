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


"""Sensor and action boundaries of the local LIBERO geometry planner."""

from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.geometry import (
    LiberoGeometryObservations,
    PlannerGeometry,
    _opening_center,
    planner_measurement,
    view_geometry_state,
)
from robots.libero.geometry_pick_place import (
    PickPlaceRequest,
    Unresolved,
    _locate,
    _run,
)
from rpent.session import EnvState
from rpent.tools import ToolResult


def test_planner_only_receives_accepted_coordinates_and_preserves_warnings():
    full = {
        "status": "partial",
        "camera": "wrist",
        "step": 4,
        "world_xyz": [9, 9, 9],
        "geometry_artifact": "geometry_test.json",
        "input_refs": {"rgb": "/evidence"},
        "geometry": {
            "rectangle_center": {
                "value": [1, 2, 3],
                "fit_accepted": True,
                "source_step": 4,
            },
            "top_corners": {
                "value": [[4, 5, 6]],
                "fit_accepted": False,
                "warnings": ["occluded"],
            },
        },
    }
    result = planner_measurement(full)
    assert result["geometry"]["rectangle_center"]["value"] == [1, 2, 3]
    rejected = result["geometry"]["top_corners"]
    assert "value" not in rejected and rejected["warnings"] == ["occluded"]
    assert "world_xyz" not in result and "input_refs" not in result
    assert result["geometry_artifact"] == full["geometry_artifact"]
    assert result["missing_geometry"] == ["top_corners"]
    rejected_only = planner_measurement(
        {
            **full,
            "status": "warning",
            "geometry": {"top_corners": full["geometry"]["top_corners"]},
        }
    )
    assert (
        rejected_only["status"] == "insufficient_evidence"
        and "world_xyz" not in rejected_only
    )
    assert rejected_only["missing_geometry"] == ["top_corners"]
    assert full["geometry"]["top_corners"]["value"] == [[4, 5, 6]]
    unresolved = planner_measurement({**full, "status": "need_observation"})
    assert all("value" not in field for field in unresolved["geometry"].values())
    failed = planner_measurement(
        {**full, "status": "error", "error": "evidence could not be saved"}
    )
    assert failed["status"] == "error" and failed["error"]
    assert "world_xyz" not in failed
    assert all("value" not in field for field in failed["geometry"].values())


def test_back_project_measures_declared_raster_without_claiming_identity():
    from toolkits.back_project import back_project
    from toolkits.geometry.observation import GeometryObservation

    h, w = 24, 32
    rr, cc = np.indices((h, w))
    xyz = np.stack((cc / 100, 1 + rr / 100, np.zeros_like(rr)), axis=-1)
    valid = np.ones((h, w), bool)
    valid[3, 5] = False
    obs = GeometryObservation(
        "overhead",
        0,
        "rotated",
        np.zeros((h, w, 3), np.uint8),
        world_xyz=xyz,
        valid_mask=valid,
        up_direction=np.array([0.0, 1.0, 0.0]),
    )
    source = SimpleNamespace(get_observation=lambda **_: obs)

    def run(selection):
        return back_project(
            camera="overhead", step=0, selection=selection, observations=source
        ).data

    result = run({"image_shape": [12, 16], "points_rc": [[1, 2], [3, 4]]})
    assert result["status"] == "sampled" and "world_xyz" not in result
    assert result["samples_xyz"] == [None, xyz[7, 9].tolist()]
    points = {"image_shape": [12, 16], "points_rc": [[1, 2], [3, 4], [5, 6]]}
    filtered = run({**points, "height_m": [1.07, 1.07]})
    assert filtered["samples_xyz"] == [None, xyz[7, 9].tolist(), None]
    assert filtered["n_height_filtered"] == 1
    empty_points = run({**points, "height_m": [2, None]})
    assert empty_points["reason"] == "no_selected_points_in_height_interval"
    assert empty_points["status"] == "need_observation"
    assert empty_points["samples_xyz"] == [None, None, None]
    assert "low <= high" in run({**points, "height_m": [2, 1]})["error"]
    for bad, diagnostic in (
        ({"points_rc": [[1, 2], [1, 16]]}, "[1, 16] at index 1"),
        ({"box_rc": [4, 4, 13, 16]}, "bottom <= 12"),
    ):
        invalid = run({"image_shape": [12, 16], **bad})
        assert diagnostic in invalid["error"] and "[12, 16]" in invalid["error"]
        assert "surface_xyz" not in invalid and "samples_xyz" not in invalid
    sampled = run(
        {"image_shape": [24, 32], "box_rc": [2, 4, 12, 20], "height_m": [1.06, 1.08]}
    )
    assert sampled["n_valid"] == 48 and 1.06 <= sampled["surface_xyz"][1] <= 1.08
    assert run({"image_shape": [12, 12], "points_rc": [[1, 1]]})["status"] == "error"
    empty = run(
        {"image_shape": [24, 32], "box_rc": [2, 4, 12, 20], "height_m": [3.0, 4.0]}
    )
    assert empty["status"] == "need_observation" and "surface_xyz" not in empty


def test_preview_and_saved_image_label_their_own_pixel_grids(tmp_path):
    state = EnvState(tmp_path)
    with state.record_step(state={}) as step:
        state.save("agentview_high.png", np.zeros((600, 800, 3), np.uint8), step=step)
    view = view_geometry_state(state, step).data["views"][0]
    assert view["image_shape"] == [384, 512]
    assert view["full_image_shape"] == [600, 800]
    assert view["full_image"] == "agentview_high.png"


def test_pinned_wrist_sample_uses_its_own_calibration(tmp_path):
    state = EnvState(tmp_path)
    for x in (0.0, 0.1):
        pose = np.eye(4)
        pose[0, 3] = x
        with state.record_step(state={}) as step:
            state.save("wrist_high.png", np.zeros((8, 8, 3), dtype=np.uint8), step=step)
            xyz = np.ones((8, 8, 3))
            state.save("wrist_world_high.npz", xyz, step=step)
            state.save(
                "wrist_metadata.json",
                {
                    "width": 4,
                    "height": 4,
                    "intrinsic_K": [[4, 0, 2], [0, 4, 2], [0, 0, 1]],
                    "extrinsic_cam2world": pose.tolist(),
                },
                step=step,
            )
    obs = LiberoGeometryObservations(state)
    historical = obs.get_observation(camera="wrist", step=0)
    latest = obs.get_observation(camera="wrist", step=-1)
    assert historical.camera_to_world[0, 3] == 0.0
    assert latest.camera_to_world[0, 3] == 0.1
    assert historical.world_frame_id == latest.world_frame_id
    np.testing.assert_array_equal(latest.intrinsics, [[8, 0, 4], [0, 8, 4], [0, 0, 1]])
    with pytest.raises(ValueError, match="calibrated RGB-D pair"):
        obs.get_observation(camera="agentview", step=0)


def test_planner_reuses_terminal_evidence_only_at_the_original_step(tmp_path):
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    calls = []

    def measure(query, camera, measurement, step, required_geometry):
        calls.append(step)
        data = {
            "status": "insufficient_evidence",
            "reason": "target_not_identified",
            "step": step,
            "view_evidence": {
                "attempts": [{"camera": camera}, {"camera": "side"}],
                "skipped": [],
            },
        }
        state.save("geometry_result.json", data, step=step)
        return ToolResult(data={**data, "geometry_artifact": "geometry_result.json"})

    planner = PlannerGeometry(SimpleNamespace(geometry=measure), state)
    request = {
        "query": "small package",
        "camera": "front",
        "measurement": "surface_point",
        "step": -1,
    }
    result = planner.geometry(**request).data
    assert result["status"] == "insufficient_evidence" and result["views_checked"] == [
        "front",
        "side",
    ]
    assert "world_xyz" not in result and "observation_request" not in result
    assert planner.geometry(**request).data == result and calls == [0]
    with state.record_step(state={}):
        pass
    assert planner.geometry(**request).data["step"] == 1 and calls == [0, 1]


def test_location_retry_uses_open_gripper_then_a_new_wrist_frame():
    request = PickPlaceRequest("small package", "rectangle_center", max_inspections=1)
    initial = {"step": 2, "state": {"robot0_eef_pos": [0, 0, 0.25]}}
    locate = _locate(request, initial, iter(range(1)))
    assert next(locate)[0] == "geometry"
    assert (
        locate.send({"status": "need_observation", "world_xyz": [0, 0, 0]})[0]
        == "inspection_pose"
    )
    move, args = locate.send({"inspection_xyz": [0.05, 0, 0.3], "source_step": 2})
    assert move == "move_to" and args["gripper"] == -1
    moved = {"step": 3, "state": {"robot0_eef_pos": [0.05, 0, 0.3]}}
    assert locate.send(moved)[0] == "view_env_state"
    name, args = locate.send(moved)
    assert (name, args["camera"], args["step"]) == ("geometry", "wrist", 3)
    stale = {
        "value": [1, 2, 3],
        "fit_accepted": True,
        "source_step": 2,
        "source_camera": "wrist",
    }
    with pytest.raises(Unresolved, match="requested capture"):
        locate.send({"status": "usable", "geometry": {"rectangle_center": stale}})


def test_finite_but_unaccepted_coordinate_never_triggers_grasp():
    routine = _locate(
        PickPlaceRequest("small package", "rectangle_center", max_inspections=0),
        {"step": 0},
        iter(()),
    )
    assert next(routine)[0] == "geometry"
    with pytest.raises(Unresolved):
        routine.send(
            {
                "status": "warning",
                "geometry": {
                    "rectangle_center": {"value": [0, 0, 0], "fit_accepted": False}
                },
            }
        )


def test_drop_point_requires_a_surrounding_lip_and_a_recess():
    phi = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    pts = np.stack(
        [0.08 * np.cos(phi) + 0.3, 0.07 * np.sin(phi) - 0.2, np.full_like(phi, 0.12)],
        axis=1,
    )
    interior = np.array(
        [
            [0.3 + x, -0.2 + y, 0.025]
            for x in [-0.02, -0.01, 0.01, 0.02]
            for y in [-0.02, -0.01, 0.01, 0.02]
        ]
    )
    np.testing.assert_allclose(
        _opening_center(np.vstack([pts, interior])), [0.3, -0.2, 0.12], atol=0.002
    )
    assert _opening_center(pts) is None
    assert _opening_center(np.vstack([pts[:140], interior])) is None


def test_recipient_reobservation_invalidates_the_old_payload_point():
    request = PickPlaceRequest("small package", "rectangle_center", max_inspections=1)
    runner = _run(request)
    assert next(runner)[0] == "view_env_state"
    assert (
        runner.send({"step": 0, "state": {"robot0_eef_pos": [0, 0, 0.25]}})[1]["query"]
        == request.target
    )
    center = {
        "value": [0, 0, 0.02],
        "fit_accepted": True,
        "source_step": 0,
        "source_camera": "agentview",
    }
    assert (
        runner.send({"status": "usable", "geometry": {"rectangle_center": center}})[1][
            "query"
        ]
        == request.destination
    )
    assert runner.send({"status": "need_observation"})[0] == "inspection_pose"
    assert (
        runner.send({"inspection_xyz": [0.10, 0, 0.25], "source_step": 0})[1]["gripper"]
        == -1
    )
    moved = {"step": 1, "state": {"robot0_eef_pos": [0.10, 0, 0.25]}}
    assert runner.send(moved)[0] == "view_env_state"
    assert runner.send(moved)[1]["camera"] == "wrist"
    assert (
        runner.send({"status": "usable", "geometry_artifact": "confirmed.json"})[0]
        == "placement_opening"
    )
    next_tool, args = runner.send(
        {"status": "usable", "camera": "wrist", "step": 1, "world_xyz": [0.1, 0, 0.15]}
    )
    assert (next_tool, args["query"], args["step"]) == ("geometry", request.target, 1)
    with pytest.raises(Unresolved):
        runner.send({"status": "need_observation"})
