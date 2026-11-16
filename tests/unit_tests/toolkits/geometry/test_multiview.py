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

"""Cross-view regressions for evidence, instance identity and pinned camera semantics."""

import numpy as np
import pytest

from rpent.tools import ToolResult
from toolkits.geometry.multiview import observe_with_alternatives
from toolkits.geometry.observation import GeometryObservation


def _obs(camera, x=0, frame="table", step=0):
    cam = np.eye(4)
    cam[0, 3] = x
    return GeometryObservation(
        camera,
        step,
        frame,
        np.zeros((8, 8, 3), np.uint8),
        depth_m=np.ones((8, 8)),
        intrinsics=np.eye(3),
        camera_to_world=cam,
        up_direction=np.array([0.0, 0.0, 1.0]),
    )


@pytest.mark.parametrize("overlap", [True, False])
def test_only_synchronized_overlapping_material_can_complete_partial_geometry(overlap):
    front = _obs("front")
    side = _obs("side", 0.4)

    class Provider:
        def get_observation(self, **kwargs):
            return front

        def additional_observations(self, first):
            assert first is front
            yield _obs("crop")
            yield _obs("foreign", 0.2, frame="other")
            yield _obs("later", 0.3, step=2)
            yield side
            raise RuntimeError("next camera is offline")

    def measure(
        query,
        camera,
        step,
        measurement,
        fields,
        cancel,
        *,
        observation,
        evidence,
        direction_camera=None,
        projected_rc=None,
    ):
        primary = observation is front
        if not primary:
            np.testing.assert_array_equal(direction_camera, front.camera_to_world)
        shift = 0.0 if primary or overlap else 1.5
        evidence["points"] = np.arange(30).reshape(10, 3) * 0.001 + shift
        quantities = {
            k: {
                "value": v,
                "fit_accepted": v is not None,
                "source_camera": camera,
                "warnings": [],
                "reason": None,
            }
            for k, v in zip(fields, ([1.0, None] if primary else [None, 2.0]))
        }
        return ToolResult(
            data={
                "camera": camera,
                "step": step,
                "query": query,
                "status": "partial",
                "found": True,
                "geometry": quantities,
                "identity_evidence": {"depth": {"accepted": True}},
            }
        )

    result = observe_with_alternatives(
        Provider(),
        measure,
        "a box",
        "front",
        0,
        "surface_point",
        ["rectangle_center", "supported_height"],
        lambda: None,
    ).data
    assert [item["reason"] for item in result["view_evidence"]["skipped"]][:3] == [
        "same_optical_view",
        "incompatible_calibration_or_world_frame",
        "different_pinned_step",
    ]
    if overlap:
        assert result["status"] == "usable"
        assert result["geometry"]["supported_height"]["source_camera"] == "side"
    else:
        assert (
            result["view_evidence"]["skipped"][-1]["reason"]
            == "additional_observation_unavailable"
        )
        assert (
            result["status"] == "partial"
            and result["geometry"]["supported_height"]["value"] is None
        )
        assert result["missing_geometry"] == ["supported_height"]
        assert "observation_request" not in result


def test_latest_observation_pins_all_cameras_to_the_resolved_step():
    front = _obs("front")
    later = _obs("later", 0.3, step=1)

    class Provider:
        def get_observation(self, **kwargs):
            return front

        def additional_observations(self, first):
            yield later

    def measure(
        query,
        camera,
        step,
        measurement,
        fields,
        cancel,
        *,
        observation,
        evidence,
        direction_camera=None,
        projected_rc=None,
    ):
        old = observation is front
        return ToolResult(
            data={
                "camera": camera,
                "step": observation.step,
                "query": query,
                "status": "partial",
                "found": True,
                "geometry": {
                    name: {"value": v, "reason": "missing", "warnings": []}
                    for name, v in zip(fields, ([1, None] if old else [None, 2]))
                },
            }
        )

    d = observe_with_alternatives(
        Provider(),
        measure,
        "a box",
        "front",
        -1,
        "surface_point",
        ["top_size", "supported_height"],
        lambda: None,
    ).data
    assert d["status"] == "partial" and d["step"] == 0
    assert d["geometry"]["top_size"]["value"] == 1
    assert d["view_evidence"]["skipped"] == [
        {"camera": "later", "reason": "different_pinned_step"}
    ]


@pytest.mark.parametrize("secondary", ["visible", "occluded", "offscreen", "ambiguous"])
def test_internal_projection_checks_the_same_step_and_visible_depth(
    tmp_path, monkeypatch, secondary
):
    from rpent.session import EnvState
    from toolkits.geometry.geometry_tool import GeometryTool

    front = _obs("front")
    side = _obs("side", 12.0 if secondary == "offscreen" else 1.0)
    if secondary == "occluded":
        depth = np.ones((8, 8))
        depth[3, 3] = 0.6
        side = GeometryObservation(
            "side",
            0,
            "table",
            side.rgb,
            depth_m=depth,
            intrinsics=side.intrinsics,
            camera_to_world=side.camera_to_world,
        )

    class Provider:
        def get_observation(self, *, camera, step):
            assert (camera, step) == ("front", -1)
            return front

        def additional_observations(self, primary):
            assert primary is front
            yield side

    calls = []

    def measure(
        query,
        camera,
        step,
        measurement,
        fields,
        cancel,
        *,
        observation,
        evidence,
        projected_rc=None,
        direction_camera=None,
    ):
        calls.append(camera)
        tool._index += 1
        assert step == 0 and query == "striped bowl"
        if camera == "front":
            assert projected_rc is None
            return ToolResult(
                data={
                    "status": "need_observation",
                    "reason": "another_visually_compatible_instance",
                    "found": False,
                    "inspection_anchor": {"xyz": front.world_xyz[3, 4].tolist()},
                    "step": step,
                    "query": query,
                    "camera": camera,
                }
            )
        assert observation is side and projected_rc == [3, 3]
        if secondary == "ambiguous":
            return ToolResult(
                data={
                    "status": "need_observation",
                    "reason": "another_visually_compatible_instance",
                    "step": step,
                    "camera": camera,
                    "query": query,
                    "found": False,
                }
            )
        return ToolResult(
            data={
                "status": "usable",
                "found": True,
                "world_xyz": side.world_xyz[3, 3].tolist(),
                "identity_evidence": {
                    "depth": {"accepted": True, "relation_verified": True}
                },
                "step": step,
                "camera": camera,
                "query": query,
            }
        )

    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass
    tool = GeometryTool(
        state=state, observations=Provider(), grounder=object(), object_first=True
    )
    monkeypatch.setattr(tool, "_single_view", measure)
    result = tool.geometry("striped bowl", "front").data
    assert state.latest_step == 0
    assert state.load(result["geometry_artifact"], step=0)["status"] == result["status"]
    if secondary == "visible":
        assert (
            calls == ["front", "side"]
            and result["status"] == "usable"
            and result["camera"] == "side"
        )
        np.testing.assert_allclose(result["world_xyz"], front.world_xyz[3, 4])
    else:
        assert result["status"] == "insufficient_evidence"
        assert (
            not result.get("world_xyz")
            and "inspection_anchor" not in result
            and "observation_request" not in result
        )
        if secondary == "ambiguous":
            assert (
                calls == ["front", "side"]
                and result["reason"] == "another_visually_compatible_instance"
            )
        else:
            reason = (
                "source_region_outside_view"
                if secondary == "offscreen"
                else "source_region_not_visible_at_measured_depth"
            )
            assert calls == ["front"] and result["view_evidence"]["skipped"] == [
                {"camera": "side", "reason": reason}
            ]


@pytest.mark.parametrize("relation_verified", [False, True])
def test_ambiguous_primary_requires_relation_at_observed_projection(relation_verified):
    front, side = _obs("front"), _obs("side", 1.0)

    class Provider:
        def get_observation(self, *, camera, step):
            return front

        def additional_observations(self, primary):
            yield side

    def measure(
        query,
        camera,
        step,
        measurement,
        fields,
        cancel,
        *,
        observation,
        evidence,
        projected_rc=None,
        direction_camera=None,
    ):
        data = {"camera": camera, "step": step, "query": query}
        if observation is front:
            return ToolResult(
                data={
                    **data,
                    "status": "need_observation",
                    "found": False,
                    "reason": "another_visually_compatible_instance",
                    "inspection_anchor": {"xyz": front.world_xyz[3, 4].tolist()},
                }
            )
        assert projected_rc == [3, 3]
        return ToolResult(
            data={
                **data,
                "status": "usable",
                "found": True,
                "world_xyz": side.world_xyz[3, 3].tolist(),
                "identity_evidence": {
                    "depth": {"accepted": True, "relation_verified": relation_verified}
                },
            }
        )

    result = observe_with_alternatives(
        Provider(),
        measure,
        "bowl between the plate and the cup",
        "front",
        0,
        "surface_point",
        None,
        lambda: None,
    ).data
    if relation_verified:
        assert result["status"] == "usable" and result["camera"] == "side"
        np.testing.assert_equal(result["world_xyz"], front.world_xyz[3, 4])
    else:
        assert result["status"] == "insufficient_evidence" and not result.get(
            "world_xyz"
        )
        assert result["reason"] == "another_visually_compatible_instance"


def test_unconfirmed_model_proposal_does_not_bind_an_independent_camera():
    from types import SimpleNamespace

    front, side = _obs("front"), _obs("side", 1.0)
    provider = SimpleNamespace(
        get_observation=lambda **_: front, additional_observations=lambda _: [side]
    )

    def measure(
        query, camera, step, measurement, fields, cancel, *, projected_rc=None, **kwargs
    ):
        assert projected_rc is None
        if camera == "front":
            return ToolResult(
                data={
                    "query": query,
                    "camera": camera,
                    "step": step,
                    "status": "need_observation",
                    "found": False,
                    "reason": "package_label_or_uniqueness_unresolved",
                    "inspection_anchor": {"xyz": front.world_xyz[3, 4].tolist()},
                }
            )
        return ToolResult(
            data={
                "query": query,
                "camera": camera,
                "step": step,
                "status": "usable",
                "found": True,
                "world_xyz": side.world_xyz[6, 6].tolist(),
            }
        )

    result = observe_with_alternatives(
        provider,
        measure,
        "cream cheese box",
        "front",
        0,
        "surface_point",
        None,
        lambda: None,
    ).data
    assert result["status"] == "usable" and result["camera"] == "side"
    np.testing.assert_equal(result["world_xyz"], side.world_xyz[6, 6])


def test_nearly_diagonal_square_keeps_metric_size_but_withholds_camera_order():
    from toolkits.geometry.measured_rectangle import rectangle_value

    face = {
        "center": np.zeros(3),
        "axes": np.eye(3)[:, :2],
        "sizes": np.array([0.05, 0.05]),
        "pitch": 0.001,
    }

    def camera(angle):
        c, s = np.cos(angle), np.sin(angle)
        t = np.eye(4)
        t[:3, :3] = [[c, 0, -s], [s, 0, c], [0, -1, 0]]
        return t

    assert rectangle_value(face, "top_corners", camera(np.deg2rad(44.7))) is None
    assert rectangle_value(face, "top_size", camera(np.deg2rad(44.7))) == [0.05, 0.05]
    assert np.shape(rectangle_value(face, "top_corners", camera(np.deg2rad(20)))) == (
        4,
        3,
    )
