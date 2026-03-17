"""
Allegro Hand grasping simulation in Isaac Sim + real-time Allegro -> Shadow Hand retargeting.

- Allegro: PD-driven, linearly interpolated over 120 frames from the initial pose to grasp_1.
- Shadow: kinematic control; each frame, target points are obtained from Allegro fingertip/middle FK,
  then joint angles are solved via position retargeting and written to the articulation.
- The first 120 frames are throttled to approximately 12 seconds; fingertip trajectories are written to fingertip_coords.csv.

Usage: python.bat replay_allegro_grasp_isaacsim.py
"""

import argparse
import csv
import os
import sys
import time

# CLI and simulation config: headless mode, Allegro model directory, grasp YAML path
CONFIG = {"headless": False}
parser = argparse.ArgumentParser()
parser.add_argument("--hand-dir", type=str, default=None, help="Allegro hand model directory (containing .usd file)")
parser.add_argument("--yaml", type=str, default="003.yaml", help="path to grasp YAML file")
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()
if args.headless:
    CONFIG["headless"] = True

from isaacsim import SimulationApp
simulation_app = SimulationApp(CONFIG)

import numpy as np
import yaml

# Script directory and asset paths: used to import dex_retargeting and the Allegro/Shadow models and configs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
_src = os.path.join(SCRIPT_DIR, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

DEFAULT_HAND_DIR = os.path.join(SCRIPT_DIR, "AllegroHand")
HAND_DIR = os.path.abspath(args.hand_dir) if args.hand_dir else DEFAULT_HAND_DIR
YAML_PATH = os.path.join(SCRIPT_DIR, args.yaml) if not os.path.isabs(args.yaml) else args.yaml

ROBOT_HANDS_DIR = os.path.join(SCRIPT_DIR, "assets", "robots", "robots", "hands")
SHADOW_CONFIG_PATH = os.path.join(SCRIPT_DIR, "src", "dex_retargeting", "configs", "offline", "shadow_hand_right.yml")
ALLEGRO_CONFIG_PATH = os.path.join(SCRIPT_DIR, "src", "dex_retargeting", "configs", "offline", "allegro_hand_right.yml")


def load_grasp_yaml(path):
    """Loads the grasp YAML file from the given path and returns the parsed dictionary (containing grasps, cspace_position, etc.)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_cspace(cspace_dict, dof_names):
    """Converts the cspace_position dict from the YAML (joint name -> angle) into a joint angle array aligned with dof_names order, in radians."""
    q = np.zeros(len(dof_names), dtype=np.float32)
    for i, name in enumerate(dof_names):
        q[i] = float(cspace_dict.get(name, 0.0))
    return q


def quat_wxyz_to_rotation_matrix(q):
    """Converts a quaternion (W, X, Y, Z) to a 3x3 rotation matrix for subsequent point transformations."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def transform_points_to_world(positions_local, world_pos, world_quat_wxyz):
    """Transforms a set of points (N,3) from the local frame to the world frame using the robot's world pose (position world_pos and quaternion world_quat_wxyz in WXYZ order). Returns (N,3). Used to convert Allegro/Shadow fingertip coordinates to world frame for CSV writing."""
    R = quat_wxyz_to_rotation_matrix(world_quat_wxyz)
    t = np.asarray(world_pos, dtype=np.float32).reshape(3)
    return (positions_local @ R.T) + t


def set_hand_drive_gains(stage, root_path, stiffness=80.0, damping=10.0, create_if_missing=False):
    """
    Iterates over all joints under the hand USD and sets the angular drive (DriveAPI) stiffness and damping.
    When create_if_missing=True, automatically creates a DriveAPI for joints that have a RevoluteJoint but
    no DriveAPI yet, enabling subsequent use of set_drive_targets_via_usd to set targets.
    Allegro uses smaller stiffness/damping for smoother grasping; Shadow is set to 0 for pure kinematic
    control (joint positions are set directly each frame).
    """
    from pxr import UsdPhysics, Usd
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return 0
    count = 0
    for prim in Usd.PrimRange(root_prim):
        try:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive or not drive.GetStiffnessAttr():
                if create_if_missing and prim.HasAPI(UsdPhysics.RevoluteJointAPI):
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                elif create_if_missing:
                    joint = UsdPhysics.RevoluteJoint(prim)
                    if joint and joint.GetAxisAttr():
                        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    else:
                        continue
                else:
                    continue
            if drive.GetStiffnessAttr():
                drive.GetStiffnessAttr().Set(float(stiffness))
            else:
                drive.CreateStiffnessAttr(float(stiffness))
            if drive.GetDampingAttr():
                drive.GetDampingAttr().Set(float(damping))
            else:
                drive.CreateDampingAttr(float(damping))
            if not drive.GetTargetPositionAttr():
                drive.CreateTargetPositionAttr(0.0)
            count += 1
        except Exception:
            pass
    return count


def set_drive_targets_via_usd(stage, root_path, target_dict_rad):
    """
    Sets the DriveAPI target angle for each joint in the hand USD by joint name. target_dict_rad is
    {joint_name: radians}; internally converted to degrees before writing to USD.
    Supports joint name variants: if a prim is named robot0_WRJ1 etc., the prefix is stripped before
    matching against the dictionary, for compatibility with Isaac Sim naming conventions.
    """
    from pxr import UsdPhysics, Usd
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return 0
    dict_short = {k.replace("_joint", ""): v for k, v in target_dict_rad.items()}
    count = 0
    for prim in Usd.PrimRange(root_prim):
        name = prim.GetName()
        val = target_dict_rad.get(name) or dict_short.get(name)
        # If direct match fails, try stripping the "robot0_" prefix from the USD name and match again
        if val is None:
            stripped = name.split("_", 1)[-1] if "_" in name else name
            val = target_dict_rad.get(stripped) or dict_short.get(stripped)
        if val is None:
            continue
        try:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                attr = drive.GetTargetPositionAttr()
                if attr:
                    attr.Set(float(np.degrees(val)))
                    count += 1
        except Exception:
            pass
    return count


def _apply_collision_and_preset(prim):
    """Applies CollisionAPI to the given USD prim so the physics engine can detect collisions; also applies PhysxCollisionAPI if PhysxSchema is available in the environment. Used for the ground plane, cube, etc."""
    from pxr import UsdPhysics
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    for physx_module in ("physxschema", "PhysxSchema"):
        try:
            mod = __import__(physx_module)
            if hasattr(mod, "PhysxCollisionAPI") and not prim.HasAPI(mod.PhysxCollisionAPI):
                mod.PhysxCollisionAPI.Apply(prim)
            break
        except Exception:
            continue


def main():
    """
    Main flow: builds the Isaac Sim scene, loads the Allegro Hand and Shadow Hand, drives the Allegro
    hand to grasp_1 as defined in the YAML, retargets the Allegro fingertip/middle positions to the Shadow
    hand each frame via position retargeting, and records fingertip trajectories to a CSV file.
    """
    import omni.usd
    from pxr import UsdGeom, Gf, Vt, UsdLux, UsdPhysics, Usd

    # Get the current USD stage; all subsequent scene, hand, and drive operations act on this stage
    stage = omni.usd.get_context().get_stage()

    # ---------- 0) Prepare the Shadow Hand retargeting solver (core of Allegro -> Shadow motion mapping) ----------
    shadow_retargeting = None
    if os.path.isdir(ROBOT_HANDS_DIR) and os.path.isfile(SHADOW_CONFIG_PATH):
        try:
            try:
                import anytree
            except ImportError:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "anytree>=2.12.0", "-q"])
                import anytree
            from dex_retargeting.retargeting_config import RetargetingConfig
            RetargetingConfig.set_default_urdf_dir(ROBOT_HANDS_DIR)
            override = dict(add_dummy_free_joint=True)
            shadow_config = RetargetingConfig.load_from_file(SHADOW_CONFIG_PATH, override=override)
            shadow_retargeting = shadow_config.build()
            shadow_joint_names = shadow_retargeting.joint_names
        except Exception:
            pass

    # ---------- 1) Scene: /World root, ground mesh, cube, ambient light ----------
    if not stage.GetPrimAtPath("/World").IsValid():
        world_xf = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world_xf.GetPrim())

    # Ground: large flat mesh with collision enabled for physics simulation
    ground_path = "/World/ground"
    if not stage.GetPrimAtPath(ground_path).IsValid():
        half = 5.0
        plane = UsdGeom.Mesh.Define(stage, ground_path)
        plane.CreatePointsAttr(Vt.Vec3fArray([
            (-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)
        ]))
        plane.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
        plane.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
        plane.CreateExtentAttr(Vt.Vec3fArray([(-half, -half, 0), (half, half, 0)]))
    ground_prim = stage.GetPrimAtPath(ground_path)
    if ground_prim and ground_prim.IsValid():
        _apply_collision_and_preset(ground_prim)

    # Cube: small cube with rigid body and mass, serves as a graspable object
    cube_path = "/World/Cube"
    if not stage.GetPrimAtPath(cube_path).IsValid():
        cube_prim = UsdGeom.Cube.Define(stage, cube_path)
        cube_prim.CreateSizeAttr(2.0)
        xform_api = UsdGeom.XformCommonAPI(cube_prim)
        xform_api.SetTranslate((0.0, 0.0, 0.05))
        xform_api.SetScale((0.05, 0.05, 0.05))
    cube_prim = stage.GetPrimAtPath(cube_path)
    if cube_prim and cube_prim.IsValid():
        if not cube_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(cube_prim)
        if not cube_prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(cube_prim)
            mass_api.CreateMassAttr(0.1)
        _apply_collision_and_preset(cube_prim)

    # Ambient light: DomeLight to illuminate the scene
    dome_path = "/World/DomeLight"
    if not stage.GetPrimAtPath(dome_path).IsValid():
        dome = UsdLux.DomeLight.Define(stage, dome_path)
        dome.CreateIntensityAttr(1000.0)

    # ---------- 4) Load Allegro Hand USD and reference it at /World/AllegroHand ----------
    usd_candidates = [
        os.path.join(HAND_DIR, "allegro_hand.usd"),
        os.path.join(HAND_DIR, "AllegroHand.usd"),
        os.path.join(HAND_DIR, "allegro.usd"),
    ]
    usd_path = None
    for p in usd_candidates:
        if os.path.isfile(p):
            usd_path = os.path.abspath(p)
            break
    if not usd_path and os.path.isdir(HAND_DIR):
        for name in sorted(os.listdir(HAND_DIR)):
            if name.endswith(".usd") or name.endswith(".usda"):
                usd_path = os.path.abspath(os.path.join(HAND_DIR, name))
                break
    if not usd_path:
        simulation_app.close()
        return

    hand_prim_path = "/World/AllegroHand"
    ref_prim = stage.OverridePrim(hand_prim_path)
    ref_prim.GetReferences().AddReference(usd_path)

    # ---------- 4b) Shadow Hand: prefer the USD in the project's ShadowHand directory; fall back to importing from URDF ----------
    shadow_prim_path = "/World/ShadowHand"
    shadow_robot = None
    shadow_usd_dir = os.path.join(SCRIPT_DIR, "ShadowHand")
    shadow_usd_path = None
    if os.path.isdir(shadow_usd_dir):
        for name in sorted(os.listdir(shadow_usd_dir)):
            if name.endswith(".usd") or name.endswith(".usda"):
                shadow_usd_path = os.path.abspath(os.path.join(shadow_usd_dir, name))
                break
    if shadow_usd_path and os.path.isfile(shadow_usd_path):
        try:
            stage.OverridePrim(shadow_prim_path)
            shadow_ref = stage.GetPrimAtPath(shadow_prim_path)
            shadow_ref.GetReferences().AddReference(shadow_usd_path)
        except Exception:
            pass
    if not shadow_usd_path or not os.path.isfile(shadow_usd_path):
        # No Shadow USD available; use Isaac Sim's URDF importer to import the Shadow model into the scene
        shadow_urdf = os.path.join(ROBOT_HANDS_DIR, "shadow_hand", "shadow_hand_right_glb.urdf")
        if os.path.isfile(shadow_urdf):
            try:
                from isaacsim.asset.importer.urdf import _urdf
                import_config = _urdf.ImportConfig()
                import_config.merge_fixed_joints = False
                import_config.fix_base = True
                import_config.import_inertia_tensor = False
                import_config.distance_scale = 1.0
                import_config.density = 0.0
                asset_root = os.path.dirname(shadow_urdf)
                asset_name = os.path.basename(shadow_urdf)
                urdf_robot = _urdf.parse_urdf(asset_root, asset_name, import_config)
                _urdf.import_robot(asset_root, asset_name, urdf_robot, import_config, stage, shadow_prim_path)
            except Exception:
                pass

    # ---------- 5) Allegro world pose (translation, rotation, scale) and joint drive gains ----------
    HAND_POSITION = [0, 0.13, 0.19]
    HAND_ORIENTATION_QUAT = [-0.65328, -0.65328, 0.2706, -0.2706]  # quaternion W, X, Y, Z
    xform = UsdGeom.Xformable(ref_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*HAND_POSITION))
    q = Gf.Quatd(HAND_ORIENTATION_QUAT[0], Gf.Vec3d(*HAND_ORIENTATION_QUAT[1:]))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(q)
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))

    set_hand_drive_gains(stage, hand_prim_path, stiffness=25.0, damping=5.0)

    # ---------- 6) Create the physics world, register Allegro/Shadow as Articulations; dof_names can only be safely read after reset ----------
    from isaacsim.core.api.physics_context import PhysicsContext
    PhysicsContext()

    try:
        from omni.isaac.core import World as IsaacWorld
        from omni.isaac.core.robots import Robot
    except ImportError:
        from isaacsim.core.api import World as IsaacWorld
        from isaacsim.core.api.robots import Robot

    world = IsaacWorld(stage_units_in_meters=1.0)
    robot = world.scene.add(Robot(prim_path=hand_prim_path, name="allegro_hand"))
    shadow_world_pos = np.array([0.0, -0.5, 0.2], dtype=np.float32)   # Shadow position in world frame, used for CSV writing
    shadow_world_rot = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32) # Shadow orientation in world frame as quaternion (WXYZ)
    shadow_art_view = None   # Shadow's ArticulationView, used in main loop to call set_joint_positions
    shadow_dof_names = []    # Shadow DOF name list in Isaac Sim, used to map retargeting joint order
    shadow_robot = None     # Read shadow_dof_names from this object after reset
    if stage.GetPrimAtPath(shadow_prim_path).IsValid():
        try:
            shadow_robot = world.scene.add(Robot(prim_path=shadow_prim_path, name="shadow_hand"))
            shadow_art_view = getattr(shadow_robot, "_articulation_view", None) or getattr(shadow_robot, "articulation_view", None)
            shadow_pos = np.array([[0.0, -0.5, 0.2]], dtype=np.float32)
            shadow_rot = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
            shadow_world_pos[:] = shadow_pos.reshape(3)
            shadow_world_rot[:] = shadow_rot.reshape(4)
            if shadow_art_view is not None and hasattr(shadow_art_view, "set_world_poses"):
                shadow_art_view.set_world_poses(positions=shadow_pos, orientations=shadow_rot)
            set_hand_drive_gains(stage, shadow_prim_path, stiffness=0.0, damping=0.0, create_if_missing=True)  # kinematic control, no PD needed
        except Exception:
            pass
    world.reset()  # Articulation is only fully initialized after reset; dof_names etc. are available after this call

    if shadow_robot is not None:
        shadow_dof_names = list(shadow_robot.dof_names) if (getattr(shadow_robot, "dof_names", None) is not None) else []
        shadow_art_view = getattr(shadow_robot, "_articulation_view", None) or getattr(shadow_robot, "articulation_view", None)

    art_view = getattr(robot, "_articulation_view", None) or getattr(robot, "articulation_view", None)
    dof_names = list(robot.dof_names)
    num_dofs = len(dof_names)

    # ---------- 7) Allegro FK: use Pinocchio RobotWrapper to compute 8 key link positions; build Isaac joint order -> Pinocchio order mapping ----------
    allegro_robot = None
    allegro_link_ids = None
    allegro_link_names = None
    allegro_sim_to_pin = None  # length 16: allegro_sim_to_pin[sim_idx] = joint index in Pinocchio
    try:
        from dex_retargeting.robot_wrapper import RobotWrapper

        # Read URDF path and target_link_names from the Allegro config for the 8 FK keypoints (4 fingertips + 4 middle segments)
        allegro_urdf_rel = None
        try:
            with open(ALLEGRO_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            allegro_urdf_rel = (
                cfg.get("retargeting", {}).get("urdf_path", None) if isinstance(cfg, dict) else None
            )
        except Exception:
            allegro_urdf_rel = None

        if not allegro_urdf_rel:
            allegro_urdf_rel = os.path.join("allegro_hand", "allegro_hand_right.urdf")

        allegro_urdf_path = os.path.join(ROBOT_HANDS_DIR, allegro_urdf_rel)
        allegro_urdf_path = os.path.abspath(allegro_urdf_path)
        allegro_robot = RobotWrapper(allegro_urdf_path)

        target_link_names = None
        try:
            with open(ALLEGRO_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                target_link_names = cfg.get("retargeting", {}).get("target_link_names", None)
        except Exception:
            pass

        if target_link_names is None:
            target_link_names = [
                "link_15.0_tip",
                "link_3.0_tip",
                "link_7.0_tip",
                "link_11.0_tip",
                "link_14.0",
                "link_2.0",
                "link_6.0",
                "link_10.0",
            ]

        allegro_link_names = []
        allegro_link_ids_list = []
        for ln in target_link_names:
            try:
                lid = allegro_robot.get_link_index(ln)
                allegro_link_names.append(ln)
                allegro_link_ids_list.append(lid)
            except Exception:
                pass

        if allegro_link_ids_list:
            allegro_link_ids = np.asarray(allegro_link_ids_list, dtype=np.int32)
        else:
            allegro_link_ids = None

        # Isaac joint order differs from Pinocchio URDF order; explicit mapping here (4 dims each for index/thumb/middle/ring)
        if allegro_robot is not None and len(dof_names) == 16 and len(allegro_robot.dof_joint_names) == 16:
            allegro_sim_to_pin = [
                0, 8, 12, 4, 1, 9, 13, 5, 2, 10, 14, 6, 3, 11, 15, 7
            ]
        else:
            allegro_sim_to_pin = None
    except Exception:
        pass

    # ---------- 8) Hand world pose; read grasp_1 from YAML to get initial_q / target_q ----------
    hand_pos = np.array(HAND_POSITION, dtype=np.float32).reshape(1, 3)
    hand_rot = np.array(HAND_ORIENTATION_QUAT, dtype=np.float32).reshape(1, 4)
    if art_view is not None and hasattr(art_view, "set_world_poses"):
        art_view.set_world_poses(positions=hand_pos, orientations=hand_rot)

    if not os.path.isfile(YAML_PATH):
        simulation_app.close()
        return
    grasp_data = load_grasp_yaml(YAML_PATH)
    grasps = grasp_data.get("grasps", {})
    if "grasp_1" not in grasps:
        simulation_app.close()
        return

    g = grasps["grasp_1"]
    cspace = g.get("cspace_position", {})

    initial_q = np.zeros(num_dofs, dtype=np.float32)
    target_q = parse_cspace(cspace, dof_names)

    # ---------- 9) Teleport Allegro to all-zero initial pose; after 10 stabilization frames, set drive target to interpolation start point (main loop will interpolate per frame) ----------
    if art_view is not None:
        for fn_pos, fn_vel in [
            ("set_joint_positions", "set_joint_velocities"),
            ("set_dof_positions", "set_dof_velocities"),
        ]:
            if hasattr(art_view, fn_pos):
                try:
                    getattr(art_view, fn_pos)(initial_q.reshape(1, -1))
                    getattr(art_view, fn_vel)(np.zeros((1, num_dofs), dtype=np.float32))
                    break
                except Exception:
                    pass

    for _ in range(10):
        world.step(render=False)
    if art_view is not None and hasattr(art_view, "set_world_poses"):
        art_view.set_world_poses(positions=hand_pos, orientations=hand_rot)

    # ---------- 10) Interpolation frame count constant; drive target is updated each frame in the main loop, initialized to initial_q here ----------
    INTERP_FRAMES = 120
    drive_target_dict = {name: float(initial_q[i]) for i, name in enumerate(dof_names)}
    set_drive_targets_via_usd(stage, hand_prim_path, drive_target_dict)
    if art_view is not None:
        for method in ("set_joint_position_targets", "set_dof_position_targets"):
            if hasattr(art_view, method):
                try:
                    getattr(art_view, method)(initial_q.reshape(1, -1))
                    break
                except Exception:
                    pass

    # Shadow: build a mapping from each Isaac DOF name to its index in the retargeting-solved q_shadow, for writing q_shadow into Isaac's joint order
    retargeting_idx_for_isaac_dof = None
    if shadow_art_view is not None and shadow_dof_names and shadow_retargeting is not None:
        snames = shadow_retargeting.joint_names
        retargeting_idx_for_isaac_dof = []
        for isaac_name in shadow_dof_names:
            idx = -1
            for j, rname in enumerate(snames):
                if rname == isaac_name or isaac_name.endswith("_" + rname) or (rname in isaac_name and "dummy" not in rname):
                    idx = j
                    break
            retargeting_idx_for_isaac_dof.append(idx)

    # Open CSV; header: frame + Allegro 4-finger fingertip xyz (Allegro base frame) + Shadow 5-finger fingertip xyz (Shadow base frame)
    csv_path = os.path.join(SCRIPT_DIR, "fingertip_coords.csv")
    csv_header = [
        "frame",
        "allegro_thumb_tip_x", "allegro_thumb_tip_y", "allegro_thumb_tip_z",
        "allegro_index_tip_x", "allegro_index_tip_y", "allegro_index_tip_z",
        "allegro_middle_tip_x", "allegro_middle_tip_y", "allegro_middle_tip_z",
        "allegro_ring_tip_x", "allegro_ring_tip_y", "allegro_ring_tip_z",
        "shadow_thtip_x", "shadow_thtip_y", "shadow_thtip_z",
        "shadow_fftip_x", "shadow_fftip_y", "shadow_fftip_z",
        "shadow_mftip_x", "shadow_mftip_y", "shadow_mftip_z",
        "shadow_rftip_x", "shadow_rftip_y", "shadow_rftip_z",
        "shadow_lftip_x", "shadow_lftip_y", "shadow_lftip_z",
    ]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)

    # ---------- 11) Main loop: advance simulation and logic only after the user clicks the UI play button; when paused, only do one app update to keep the UI responsive ----------
    import omni.timeline
    from omni.kit.app import get_app
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()  # pause on startup; wait for the user to click play before starting simulation
    frame_idx = 0
    loop_start_time = time.perf_counter()
    while simulation_app.is_running():
        if not timeline.is_playing():
            get_app().update()
            continue
        frame_idx += 1

        # First INTERP_FRAMES frames: linearly interpolate Allegro drive target from initial_q to target_q, converging to grasp_1 within ~120 frames
        if frame_idx <= INTERP_FRAMES:
            t = float(frame_idx) / float(INTERP_FRAMES)
            current_target = initial_q + t * (target_q - initial_q)
            drive_target_dict = {name: float(current_target[i]) for i, name in enumerate(dof_names)}
            set_drive_targets_via_usd(stage, hand_prim_path, drive_target_dict)
            if art_view is not None:
                for method in ("set_joint_position_targets", "set_dof_position_targets"):
                    if hasattr(art_view, method):
                        try:
                            getattr(art_view, method)(current_target.reshape(1, -1))
                            break
                        except Exception:
                            pass

        world.step(render=True)
        # Each frame, force the hand world pose back to the configured value to prevent physics drift
        if art_view is not None and hasattr(art_view, "set_world_poses"):
            art_view.set_world_poses(positions=hand_pos, orientations=hand_rot)

        # Read current Allegro joint angles from simulation, convert to Pinocchio order, run FK to get 8 keypoints, then map to Shadow's 10 target points and retarget
        if art_view is not None and allegro_robot is not None and allegro_link_ids is not None and allegro_sim_to_pin is not None:
            q_allegro_sim = None
            for getter in ("get_joint_positions", "get_dof_positions"):
                if hasattr(art_view, getter):
                    try:
                        vals = getattr(art_view, getter)()
                        q_arr = np.asarray(vals, dtype=np.float32)
                        if q_arr.ndim > 1:
                            q_arr = q_arr.reshape(-1)
                        q_allegro_sim = q_arr
                        break
                    except Exception:
                        pass
            if q_allegro_sim is not None and q_allegro_sim.size == num_dofs:
                # Isaac may return values in degrees; if magnitudes are clearly larger than π, convert from degrees to radians
                if q_allegro_sim.size > 0 and np.max(np.abs(q_allegro_sim)) > np.pi + 0.5:
                    q_allegro_sim = np.deg2rad(q_allegro_sim)
                q_pin = np.array(allegro_robot.q0, copy=True)
                for sim_idx, pin_idx in enumerate(allegro_sim_to_pin):
                    if sim_idx >= len(q_allegro_sim) or pin_idx < 0 or pin_idx >= len(q_pin):
                        continue
                    q_pin[pin_idx] = float(q_allegro_sim[sim_idx])

                try:
                    allegro_robot.compute_forward_kinematics(q_pin)
                    keypoints = []
                    for lid in allegro_link_ids:
                        pose = allegro_robot.get_link_pose(int(lid))
                        pos = np.asarray(pose[:3, 3], dtype=np.float32)  # translation part of the 4x4 pose matrix
                        keypoints.append(pos)
                    allegro_keypoints = np.stack(keypoints, axis=0) if keypoints else None
                except Exception:
                    pass

                # Fill Allegro's 8 points (4 fingertips + 4 middle segments) into Shadow's 10 target points (5 fingertips + 5 middle segments; little finger reuses ring finger)
                if allegro_keypoints is not None and shadow_retargeting is not None and allegro_keypoints.shape[0] == 8:
                    # Rotate -90° about Z to transform from Allegro frame to Shadow frame (commented out; using Allegro frame directly as target)
                    # R_allegro_to_shadow = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)  # rotate -90° about Z
                    # allegro_in_shadow_frame = (allegro_keypoints @ R_allegro_to_shadow.T).astype(np.float32)
                    allegro_in_shadow_frame = allegro_keypoints.astype(np.float32)  # no rotation transform applied
                    ref_value = np.zeros((10, 3), dtype=np.float32)
                    ref_value[0] = allegro_in_shadow_frame[0]   # thtip <- thumb_tip
                    ref_value[1] = allegro_in_shadow_frame[1]   # fftip <- index_tip
                    ref_value[2] = allegro_in_shadow_frame[2]   # mftip <- middle_tip
                    ref_value[3] = allegro_in_shadow_frame[3]   # rftip <- ring_tip
                    ref_value[4] = allegro_in_shadow_frame[3]   # lftip <- ring_tip (little finger reuses ring)
                    ref_value[5] = allegro_in_shadow_frame[4]   # thmiddle <- thumb_mid
                    ref_value[6] = allegro_in_shadow_frame[5]   # ffmiddle <- index_mid
                    ref_value[7] = allegro_in_shadow_frame[6]   # mfmiddle <- middle_mid
                    ref_value[8] = allegro_in_shadow_frame[7]   # rfmiddle <- ring_mid
                    ref_value[9] = allegro_in_shadow_frame[7]   # lfmiddle <- ring_mid (little finger reuses ring)
                    try:
                        q_shadow = shadow_retargeting.retarget(ref_value)  # optimizer finds joint angles that bring Shadow fingertips/middles closest to ref_value
                        # Fill shadow_positions with q_shadow in Isaac's DOF order and directly set joint positions (kinematic control)
                        if shadow_art_view is not None and retargeting_idx_for_isaac_dof is not None and len(q_shadow) == len(shadow_joint_names):
                            n_dof = len(shadow_dof_names)
                            shadow_positions = np.zeros(n_dof, dtype=np.float32)
                            for i in range(n_dof):
                                j = retargeting_idx_for_isaac_dof[i]
                                if j >= 0:
                                    shadow_positions[i] = float(q_shadow[j])
                            for method in ("set_joint_positions", "set_dof_positions"):
                                if hasattr(shadow_art_view, method):
                                    try:
                                        getattr(shadow_art_view, method)(shadow_positions.reshape(1, -1))
                                        if hasattr(shadow_art_view, "set_joint_velocities"):
                                            getattr(shadow_art_view, "set_joint_velocities")(np.zeros((1, n_dof), dtype=np.float32))
                                        elif hasattr(shadow_art_view, "set_dof_velocities"):
                                            getattr(shadow_art_view, "set_dof_velocities")(np.zeros((1, n_dof), dtype=np.float32))
                                        break
                                    except Exception:
                                        continue

                        # Compute Shadow 5-finger fingertips from the solved q_shadow; CSV records each hand in its own base frame (Allegro first 4 points are fingertips, Shadow uses all 5 fingertips)
                        shadow_robot = shadow_retargeting.optimizer.robot
                        shadow_robot.compute_forward_kinematics(q_shadow)
                        tip_indices = shadow_retargeting.optimizer.target_link_indices[:5]
                        shadow_tips = np.array([shadow_robot.get_link_pose(int(i))[:3, 3] for i in tip_indices])
                        allegro_tips_local = allegro_keypoints[:4]   # 4 fingertips in Allegro base frame
                        row = [frame_idx]
                        for i in range(4):
                            row.extend([float(allegro_tips_local[i, 0]), float(allegro_tips_local[i, 1]), float(allegro_tips_local[i, 2])])
                        for i in range(5):
                            row.extend([float(shadow_tips[i, 0]), float(shadow_tips[i, 1]), float(shadow_tips[i, 2])])
                        if frame_idx <= 120:
                            csv_writer.writerow(row)
                    except Exception:
                        pass

        # Throttle for the first 120 frames: ensure total duration of 120 frames is ~12 seconds (~0.1s per frame)
        if frame_idx <= 120:
            target_elapsed = frame_idx * (12.0 / 120)
            elapsed = time.perf_counter() - loop_start_time
            if elapsed < target_elapsed:
                time.sleep(target_elapsed - elapsed)

    csv_file.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
