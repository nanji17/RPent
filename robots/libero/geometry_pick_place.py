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


"""A bounded local RPent planner for evaluating visual reacquisition and pick-and-place."""

from __future__ import annotations

import json
import math
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic_ai.messages import RetryPromptPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel


@dataclass(frozen=True)
class PickPlaceRequest:
    target: str
    quantity: str
    destination: str = "the woven basket with a white fabric lining"
    max_inspections: int = 4

    def __post_init__(self):
        if not self.target.strip() or not self.destination.strip():
            raise ValueError("target and destination must be nonempty descriptions")
        if self.quantity not in {"rectangle_center", "lid_center"}:
            raise ValueError("pick a rectangular package or a can by its upper face")
        if not 0 <= self.max_inspections <= 4:
            raise ValueError("max_inspections must be between zero and four")


class Unresolved(RuntimeError):
    """An observation or an action did not supply the prerequisite for the next one."""


def _call(name, **args):
    reply = yield (name, args)
    if reply.get("error"):
        raise Unresolved(str(reply["error"]))
    return reply


def _point(value):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise Unresolved("no finite measured point")
    return array


def _move(start, end, grip, yaw=None):
    end = _point(end)
    n = max(1, math.ceil(np.linalg.norm(end[:2] - start[:2]) / 0.16))
    waypoints = np.linspace(start, end, n + 1)[1:]
    for waypoint in waypoints:
        opts = {"target_yaw": float(yaw)} if yaw is not None else {}
        reply = yield from _call(
            "move_to",
            xyz=waypoint.tolist(),
            gripper=float(grip),
            tol=0.003,
            step_clip=0.018,
            max_steps=70,
            **opts,
        )
        reached = _point(reply["state"]["robot0_eef_pos"])
        if reply.get("terminated") or reply.get("truncated"):
            raise Unresolved("episode ended during an approach or transport")
        if np.linalg.norm(reached - waypoint) > 0.025:
            raise Unresolved("requested waypoint was not reached")
    return reached


def _inspect(state, query, attempt, budget, reason):
    if next(budget, None) is None:
        raise Unresolved(reason)
    pose = yield from _call("inspection_pose", query=query, attempt=attempt)
    if pose.get("source_step") != state["step"]:
        raise Unresolved("inspection pose was planned from a different capture")
    current = _point(state["state"]["robot0_eef_pos"])
    reached = yield from _move(current, pose["inspection_xyz"], -1)
    fresh = yield from _call("view_env_state", step=-1)
    if fresh["step"] <= state["step"] or np.linalg.norm(reached - current) < 0.01:
        raise Unresolved("no displaced wrist observation after inspection motion")
    return fresh


def _locate(request, state, budget):
    camera = "agentview"
    query = request.target
    fields = [request.quantity]
    if request.quantity == "rectangle_center":
        fields += ["top_corners", "supported_height"]
    for attempt in range(request.max_inspections + 1):
        result = yield from _call(
            "geometry",
            query=query,
            camera=camera,
            step=state["step"],
            required_geometry=fields,
        )
        geom = result.get("geometry", {})
        field = geom.get(request.quantity, {})
        if (
            result.get("status") in ("usable", "warning", "partial")
            and field.get("value") is not None
            and field.get("fit_accepted")
        ):
            if field.get("source_step") != state["step"] or field.get(
                "source_camera"
            ) not in result.get("views_checked", [camera]):
                raise Unresolved("measurement is not from the requested capture")
            return _point(field["value"]), geom, state
        reason = result.get("reason") or "target geometry unavailable"
        if result.get("status") not in (
            "partial",
            "warning",
            "insufficient_evidence",
            "need_observation",
        ):
            raise Unresolved(reason)
        state = yield from _inspect(state, query, attempt, budget, reason)
        camera = "wrist"
    raise Unresolved("view budget exhausted")


def _destination(request, state, budget):
    camera = "agentview"
    for attempt in range(request.max_inspections + 1):
        result = yield from _call(
            "geometry",
            query=request.destination,
            camera=camera,
            step=state["step"],
            measurement="visible_region",
        )
        if result.get("status") == "usable" and result.get("geometry_artifact"):
            region_camera = result.get("camera", camera)
            if region_camera not in result.get("views_checked", [camera]):
                raise Unresolved("recipient measurement is not from a checked camera")
            result = yield from _call(
                "placement_opening",
                geometry_artifact=result["geometry_artifact"],
                step=state["step"],
            )
            if result.get("status") == "usable" and (
                result.get("camera"),
                result.get("step"),
            ) == (region_camera, state["step"]):
                return _point(result["world_xyz"]), state
        reason = result.get("reason") or "destination opening not resolved"
        if result.get("status") not in (
            "partial",
            "warning",
            "insufficient_evidence",
            "need_observation",
        ):
            raise Unresolved(reason)
        state = yield from _inspect(state, request.destination, attempt, budget, reason)
        camera = "wrist"
    raise Unresolved("view budget exhausted")


def _run(request) -> Generator:
    budget = iter(range(request.max_inspections))
    state = yield from _call("view_env_state", step=-1)
    target, geometry, source = yield from _locate(request, state, budget)
    goal, state = yield from _destination(request, source, budget)
    if state["step"] != source["step"]:
        # Approaching the recipient changed the scene. Reconfirm the payload before closing.
        target, geometry, state = yield from _locate(request, state, budget)
    pos = _point(state["state"]["robot0_eef_pos"])
    corners = geometry.get("top_corners", {})
    yaw = None
    if corners.get("fit_accepted"):
        points = np.asarray(corners["value"])[:, :2]
        _, axes = np.linalg.eigh(np.cov(points.T))
        yaw = math.atan2(axes[1, -1], axes[0, -1])
        from scipy.spatial.transform import Rotation

        frame = Rotation.from_quat(state["state"]["robot0_eef_quat"]).as_matrix()
        current = math.atan2(frame[1, 0], frame[0, 0])
        yaw = current + (yaw - current + math.pi / 2) % math.pi - math.pi / 2
    height = geometry.get("supported_height", {})
    grip_z = target[2] - (
        0.5 * height["value"] if height.get("fit_accepted") else 0.012
    )
    above = target.copy()
    above[2] = max(pos[2], target[2] + 0.15)
    pos = yield from _move(pos, above, -1, yaw)
    pos = yield from _move(pos, [*target[:2], grip_z], -1, yaw)
    close = yield from _call("set_gripper", gripper=1.0, steps=14)
    pos = _point(close["state"]["robot0_eef_pos"])
    travel = max(target[2] + 0.18, goal[2] + 0.13)
    pos = yield from _move(pos, [*pos[:2], travel], 1)
    fingers = np.abs(
        (yield from _call("view_env_state", step=-1))["state"]["robot0_gripper_qpos"]
    ).sum()
    if fingers < 0.007:
        raise Unresolved("empty closed gripper after lift")
    pos = yield from _move(pos, [*goal[:2], travel], 1)
    pos = yield from _move(pos, [*goal[:2], goal[2] + 0.065], 1)
    release = yield from _call("release", max_steps=28)
    if not release.get("terminated"):
        raise Unresolved("release did not satisfy the native task")


def pick_place_model(request: PickPlaceRequest) -> FunctionModel:
    """Drive ApiAgentLoop using its returned tool messages and current measurements."""
    routine = _run(request)

    async def respond(messages, info):
        answer = next(
            (v for v in reversed(messages[-1].parts) if isinstance(v, ToolReturnPart)),
            None,
        )
        try:
            if any(isinstance(part, RetryPromptPart) for part in messages[-1].parts):
                raise Unresolved("planner tool request was refused")
            if answer is None:
                name, args = next(routine)
            else:
                content = answer.content
                if isinstance(content, list):
                    content = content[0]
                data = json.loads(content) if isinstance(content, str) else content
                name, args = routine.send(data)
        except StopIteration:
            name, args = (
                "finish",
                {
                    "status": "success",
                    "summary": "Released after the native environment terminated.",
                },
            )
        except Unresolved as exc:
            name, args = "finish", {"status": "stuck", "summary": str(exc)}
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(args))}

    return FunctionModel(
        stream_function=respond, model_name="local-geometry-pick-place"
    )


def solve_pick_place(toolkit, request: PickPlaceRequest, task: str, dashboard_events):
    """Run one bounded local task through the standard Planner and registered robot tools."""
    from rpent.planner.api_loop import ApiAgentLoop

    planner = ApiAgentLoop(
        model=pick_place_model(request),
        dashboard_events=dashboard_events,
        no_images=True,
        timeout_s=850,
    )
    return planner.solve(
        system_prompt="Use calibrated geometry and acquire a fresh wrist view when evidence is insufficient.",
        user_message=task,
        toolkit=toolkit,
        max_turns=55,
    )


def main():
    """Connect to running LIBERO and Molmo services and execute one pick-and-place task."""
    import argparse
    from contextlib import ExitStack, closing

    from robots.libero.env_client import LiberoEnvClient
    from robots.libero.toolkit import LiberoToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.robots.components.molmo_client import MolmoClient
    from rpent.robots.components.sam3_client import Sam3Client
    from rpent.utils.logging import init_output_dir
    from rpent.utils.rpc import make_rpc_client
    from toolkits.geometry.local_point_agent import LocalPointAgent

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--env-endpoint", required=True)
    parser.add_argument("--molmo-endpoint", required=True)
    parser.add_argument(
        "--sam3-endpoint", help="SAM3 service for part masks and segmentation fallback."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--quantity", choices=("rectangle_center", "lid_center"), required=True
    )
    parser.add_argument("--destination", default=PickPlaceRequest.destination)
    parser.add_argument("--max-inspections", type=int, default=4)
    args = parser.parse_args()
    request = PickPlaceRequest(
        args.target, args.quantity, args.destination, args.max_inspections
    )
    args.output.mkdir(parents=True, exist_ok=False)
    init_output_dir(args.output)
    with ExitStack() as stack:
        rpc = stack.enter_context(closing(make_rpc_client(args.env_endpoint)))
        env = LiberoEnvClient(rpc, expected_meta=rpc.call("env.get_env_meta"))
        molmo = stack.enter_context(closing(make_rpc_client(args.molmo_endpoint)))
        grounder = LocalPointAgent(MolmoClient(molmo, point_mode="native"))
        sam3 = (
            Sam3Client(
                stack.enter_context(closing(make_rpc_client(args.sam3_endpoint)))
            )
            if args.sam3_endpoint
            else None
        )
        events = NullDashboardEventSink()
        tk = LiberoToolkit(
            runtime_kwargs={"env": env, "model": None, "sam3_client": sam3},
            geometry_grounder=grounder,
            geometry_only=sam3 is None,
            geometry_inspection_experiment=True,
            dashboard_events=events,
            memory=MemoryManager(
                root=args.output / "memory", memory_access="read_only"
            ),
            state_output_dir=args.output / "observations",
        )
        stack.callback(tk.close)
        result = solve_pick_place(tk, request, env.get_task_language(), events)
        print(
            json.dumps(
                {"finish": result.finish_result, "error": result.error},
                ensure_ascii=False,
            )
        )
        return 0 if (result.finish_result or {}).get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
