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


"""Capture RGB-D validation frames and independent reference poses in native simulators."""

import argparse
from pathlib import Path


def capture_robotwin_calibration(out):
    import json
    import os

    import numpy as np
    import sapien
    from PIL import Image
    from sapien import Pose
    from transforms3d.quaternions import mat2quat

    out.mkdir(parents=True, exist_ok=True)
    scene = sapien.Scene()
    scene.set_timestep(1 / 120)
    scene.set_ambient_light([0.5] * 3)
    scene.add_directional_light([0.4, 0.4, -1], [1.8] * 3, shadow=True)

    def box(name, halves, pos, color, still):
        b = scene.create_actor_builder()
        b.add_box_visual(
            half_size=halves,
            material=sapien.render.RenderMaterial(base_color=[*color, 1]),
        )
        b.add_box_collision(half_size=halves)
        b.initial_pose = Pose(pos)
        return b.build_static(name=name) if still else b.build(name=name)

    box("table", [0.55, 0.39, 0.04], [0, 0, 0.70], [0.52, 0.44, 0.33], True)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(
        str(Path(os.environ["TWIN_ASSETS"]) / "embodiments/franka-panda/panda.urdf")
    )
    robot.set_root_pose(Pose([-0.39, 0.22, 0.74]))
    robot.set_qpos([0, -0.5, 0, -2.1, 0, 1.7, 0.78, 0.02, 0.02])
    actors = {}
    for n, pt, col in [
        ("red", [-0.11, -0.04, 0.79], [0.83, 0.03, 0.03]),
        ("green", [0.14, 0.02, 0.80], [0.04, 0.66, 0.04]),
    ]:
        a = box(n, [0.025] * 3, pt, col, False)
        actors[n] = a
        a.set_pose(Pose(pt, [np.cos(0.22), 0, 0, np.sin(0.22)]))
    camera = scene.add_camera("head", 640, 480, np.deg2rad(48), 0.03, 5)
    eye = np.array([0.32, -0.63, 1.18])
    f = np.array([0.0, 0.0, 0.78]) - eye
    f /= np.linalg.norm(f)
    left = np.cross([0, 0, 1.0], f)
    left /= np.linalg.norm(left)
    v = np.cross(f, left)
    camera.set_pose(Pose(eye, mat2quat(np.stack((f, left, v), axis=1))))
    records = []
    for step in range(19):
        if step in (0, 6, 18):
            scene.update_render()
            camera.take_picture()
            rgb = np.asarray(camera.get_picture("Color"))[:, :, :3]
            rgb = np.uint8(np.clip(rgb * 255, 0, 255))
            positions = np.asarray(camera.get_picture("Position"))
            depth = -positions[..., 2]
            cv = np.eye(4)
            cv[:3, :4] = camera.get_extrinsic_matrix()[:3, :4]
            extrinsic = np.linalg.inv(cv)
            K = camera.get_intrinsic_matrix()
            prefix = out / f"twin-{step}"
            Image.fromarray(rgb).save(str(prefix) + ".png")
            np.savez_compressed(str(prefix) + ".npz", depth=depth, K=K, cam=extrinsic)
            for color in ("red", "green"):
                pose = actors[color].get_pose()
                records.append(
                    {
                        "id": f"twin-{step}-{color}",
                        "frame": str(prefix),
                        "query": f"the {color} cube",
                        "step": step,
                        "half": [0.025] * 3,
                        "pose": pose.to_transformation_matrix().tolist(),
                    }
                )
        scene.step()
    (out / "twin-truth.json").write_text(json.dumps(records))
    print("frames", len(records), flush=True)


def capture_robotwin_task(out, seed=21):
    """Collect head RGB-D and independently scoped actor FK in the published dual-arm task."""
    import json
    import os

    import numpy as np
    import yaml
    from PIL import Image
    from robotwin.assets import validate_root
    from robotwin.config import load_task_config
    from robotwin.envs.stack_blocks_two import stack_blocks_two

    out.mkdir(parents=True, exist_ok=True)
    assets = Path(validate_root(os.environ["ROBOTWIN_ASSETS_ROOT"])["root"])
    os.environ["ASSETS_PATH"] = str(assets)
    config = load_task_config("demo_clean")
    embodiment = config["embodiment"][0]
    robot = assets / load_task_config("_embodiment_config")[embodiment]["file_path"]
    robot_config = yaml.safe_load((robot / "config.yml").read_text())
    camera_config = load_task_config("_camera_config")[
        config["camera"]["head_camera_type"]
    ]
    config.update(
        task_name="stack_blocks_two",
        task_config="demo_clean",
        seed=seed,
        now_ep_num=0,
        planner_backend="mplib",
        left_robot_file=str(robot),
        right_robot_file=str(robot),
        left_embodiment_config=robot_config,
        right_embodiment_config=robot_config,
        dual_arm_embodied=True,
        head_camera_h=camera_config["h"],
        head_camera_w=camera_config["w"],
        save_data=False,
        collect_data=False,
        eval_video_log=False,
        render_freq=0,
        save_path=str(out),
        eval_mode=True,
        need_plan=False,
    )
    task = stack_blocks_two()
    task.setup_demo(**config)
    records = []
    try:
        for step in range(19):
            if step in (0, 6, 18):
                task._update_render()
                camera = task.cameras.static_camera_list[task.cameras.head_camera_id]
                camera.take_picture()
                rgb = np.uint8(
                    np.clip(camera.get_picture("Color")[:, :, :3] * 255, 0, 255)
                )
                depth = -camera.get_picture("Position")[:, :, 2]
                world_to_camera = np.eye(4)
                world_to_camera[:3, :4] = camera.get_extrinsic_matrix()[:3, :4]
                group = "nativeTwin" if seed == 21 else f"nativeTwin{seed}"
                prefix = out / f"{group}-{step}"
                Image.fromarray(rgb).save(str(prefix) + ".png")
                np.savez_compressed(
                    str(prefix) + ".npz",
                    depth=depth,
                    cam=np.linalg.inv(world_to_camera),
                    K=camera.get_intrinsic_matrix(),
                )
                for color, actor in (("red", task.block1), ("green", task.block2)):
                    records.append(
                        {
                            "id": f"{prefix.name}-{color}",
                            "frame": str(prefix),
                            "query": f"the {color} cube",
                            "step": step,
                            "camera": "head_camera",
                            "half": [0.025] * 3,
                            "pose": actor.get_pose()
                            .to_transformation_matrix()
                            .tolist(),
                        }
                    )
            task.scene.step()
        (out / f"{group}-truth.json").write_text(json.dumps(records))
    finally:
        task.close_env()


def capture_robocasa(out, *, varied=False, target_seed=24018):
    import json

    import mujoco
    import numpy as np
    import robocasa  # noqa: F401  # Registers the RoboCasa environments.
    import robosuite
    from PIL import Image
    from robosuite.controllers import load_composite_controller_config

    out.mkdir(parents=True, exist_ok=True)
    cams = ["robot0_agentview_left", "robot0_agentview_right"]
    env = robosuite.make(
        env_name="PickPlaceCounterToCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(
            controller=None, robot="PandaOmron"
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        layout_ids=[1],
        style_ids=[1],
        seed=17,
    )
    env.reset()
    sim = env.sim
    model = sim.model._model
    data = sim.data._data
    W, H = 640, 480
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    renderer = mujoco.Renderer(model, height=H, width=W)
    view_options = mujoco.MjvOption()
    view_options.geomgroup[0] = int(env.render_collision_mesh)
    view_options.geomgroup[1] = int(env.render_visual_mesh)
    records = []
    rng = np.random.default_rng(target_seed)
    palette = {
        "red": [0.88, 0.04, 0.02, 1],
        "green": [0.06, 0.84, 0.06, 1],
        "blue": [0.06, 0.12, 0.92, 1],
        "yellow": [0.94, 0.81, 0.04, 1],
    }
    steps = (1, 3, 5, 7, 9, 11) if varied else (0, 4, 12)
    try:
        for step in range(13):
            if step in steps:
                for view, cam_name in enumerate(cams):
                    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                    if cid < 0:
                        raise ValueError(cam_name)
                    rot = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
                    cam = np.eye(4)
                    cam[:3, :3] = rot
                    cam[:3, 3] = data.cam_xpos[cid]
                    focal = (H / 2) / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
                    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])
                    angle = 0.2 + 0.025 * step
                    half = np.array([0.033, 0.021, 0.027])
                    offset = np.array([0.04, -0.04, 0.63])
                    color = "red"
                    if varied:
                        half = rng.uniform([0.023, 0.014, 0.015], [0.048, 0.028, 0.038])
                        offset = rng.uniform([-0.17, -0.12, 0.47], [0.16, 0.10, 0.74])
                        angle = rng.uniform(-np.pi, np.pi)
                        color = tuple(palette)[len(records) % len(palette)]
                    c, s = np.cos(angle), np.sin(angle)
                    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
                    pos = cam[:3, 3] + rot @ offset

                    def prepare():
                        renderer.update_scene(
                            data, camera=cam_name, scene_option=view_options
                        )
                        scene = renderer.scene
                        g = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(
                            g,
                            mujoco.mjtGeom.mjGEOM_BOX,
                            half,
                            pos,
                            rz.ravel(),
                            np.array(palette[color], np.float32),
                        )
                        scene.ngeom += 1

                    renderer.disable_depth_rendering()
                    prepare()
                    rgb = renderer.render().copy()
                    renderer.enable_depth_rendering()
                    prepare()
                    depth = renderer.render().copy()
                    group = "varied" if varied else "casa"
                    if varied and target_seed != 24018:
                        group = f"varied{target_seed}"
                    prefix = out / f"{group}-{step}-{view}"
                    Image.fromarray(rgb).save(str(prefix) + ".png")
                    np.savez_compressed(str(prefix) + ".npz", depth=depth, K=K, cam=cam)
                    T = np.eye(4)
                    T[:3, :3] = rz
                    T[:3, 3] = pos
                    records.append(
                        {
                            "id": prefix.name,
                            "frame": str(prefix),
                            "query": f"the {color} rectangular block",
                            "step": step,
                            "half": half.tolist(),
                            "pose": T.tolist(),
                            "camera": cam_name,
                        }
                    )
            if step < 12:
                env.step(np.zeros(env.action_spec[0].shape))
        (out / f"{group}-truth.json").write_text(json.dumps(records))
        print("measurements", len(records))
    finally:
        renderer.close()
        env.close()


def capture_libero(out, cases=None):
    import json

    import numpy as np
    from liberopro.liberopro import benchmark, get_libero_path
    from liberopro.liberopro.envs import OffScreenRenderEnv
    from PIL import Image
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    from toolkits.geometry.evaluation.checklist import build_manifest
    from toolkits.geometry.evaluation.libero import REFERENCE_OBJECTS

    out.mkdir(parents=True, exist_ok=True)
    m = build_manifest()
    cases = cases or [("libero_10_task", 9, 8), ("libero_object_task", 0, 11)]
    for suite, task, seed in cases:
        bench = benchmark.get_benchmark(suite)()
        spec = bench.get_task(task)
        path = (
            Path(get_libero_path("bddl_files")) / spec.problem_folder / spec.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=str(path),
            camera_heights=512,
            camera_widths=512,
            camera_names=["agentview"],
            camera_depths=True,
        )
        rows = []
        try:
            env.seed(seed)
            env.reset()
            init = bench.get_task_init_states(task)
            env.set_init_state(init[seed % len(init)])
            action = np.zeros(7)
            action[-1] = -1
            for _ in range(10):
                env.step(action)
            sim = env.sim
            raw = env
            while hasattr(raw, "env"):
                raw = raw.env
            targets = next(
                c for c in m["cases"] if c["suite"] == suite and c["task"] == task
            )["requests"]
            for step in range(26):
                if step in (0, 8, 25):
                    rgb, d = sim.render(
                        width=512, height=512, camera_name="agentview", depth=True
                    )
                    rgb = rgb[::-1].copy()
                    depth = get_real_depth_map(sim, d)[::-1].copy()
                    K = get_camera_intrinsic_matrix(sim, "agentview", 512, 512)
                    cam = get_camera_extrinsic_matrix(sim, "agentview")
                    prefix = out / f"motion-{suite}-{task}-{seed}-{step}"
                    Image.fromarray(rgb).save(str(prefix) + ".png")
                    np.savez_compressed(str(prefix) + ".npz", K=K, cam=cam, depth=depth)
                    for r in targets:
                        ref = r["reference"]
                        i = raw.obj_body_id[REFERENCE_OBJECTS[ref]]
                        T = np.eye(4)
                        T[:3, 3] = sim.data.body_xpos[i].copy()
                        T[:3, :3] = sim.data.body_xmat[i].reshape(3, 3).copy()
                        rows.append(
                            {
                                "id": prefix.name + "-" + r["slot"],
                                "frame": str(prefix),
                                "step": step,
                                "query": m["references"][ref]["query"],
                                "ref": ref,
                                "pose": T.tolist(),
                                "camera": "agentview",
                            }
                        )
                if step < 25:
                    action[:3] = [0.20 if step < 12 else -0.18, 0.12, -0.06]
                    env.step(action)
            (out / f"motion-{suite}-{task}-{seed}-truth.json").write_text(
                json.dumps(rows)
            )
            print(suite, "frames", len(rows), flush=True)
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene",
        choices=(
            "libero",
            "robocasa",
            "robocasa-varied",
            "robotwin-calibration",
            "robotwin-task",
        ),
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--robotwin-seed", type=int, default=21)
    parser.add_argument(
        "--libero-case",
        nargs=3,
        action="append",
        metavar=("SUITE", "TASK", "SEED"),
        help="Repeat to capture selected LIBERO tasks and seeds",
    )
    parser.add_argument(
        "--target-seed",
        type=int,
        default=24018,
        help="Placement seed for robocasa-varied calibration targets",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("capture output must be new or empty")
    captures = {
        "libero": lambda path: capture_libero(
            path,
            [(suite, int(task), int(seed)) for suite, task, seed in args.libero_case]
            if args.libero_case
            else None,
        ),
        "robocasa": capture_robocasa,
        "robocasa-varied": lambda path: capture_robocasa(
            path, varied=True, target_seed=args.target_seed
        ),
        "robotwin-calibration": capture_robotwin_calibration,
        "robotwin-task": lambda path: capture_robotwin_task(path, args.robotwin_seed),
    }
    captures[args.scene](args.output)


if __name__ == "__main__":
    main()
