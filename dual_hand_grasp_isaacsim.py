"""
Dual hand grasping simulation in Isaac Sim.

- Allegro: PD-driven, linearly interpolated over 120 frames from the initial pose to grasp_1.
- Shadow: PD-driven, linearly interpolated over 120 frames from open hand to a hardcoded grasp pose.
- Both hands grasp their own cube simultaneously when the user clicks play.

Usage: python.bat dual_hand_grasp_isaacsim.py
"""

import argparse
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

# Script directory and asset paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_HAND_DIR = os.path.join(SCRIPT_DIR, "AllegroHand")
HAND_DIR = os.path.abspath(args.hand_dir) if args.hand_dir else DEFAULT_HAND_DIR
YAML_PATH = os.path.join(SCRIPT_DIR, args.yaml) if not os.path.isabs(args.yaml) else args.yaml

# Hardcoded Shadow hand grasp target joint angles (radians), matched by substring against USD joint names.
# Joint names in the USD are robot0_FFJ0..FFJ3, robot0_MFJ0..MFJ3, etc. (0-indexed).
# xJ0 = distal, xJ1 = middle, xJ2 = proximal flexion; xJ3 = side spread (±20 deg, keep at 0).
# THJ0 flexes in the negative direction (limits: -90 to 0 deg).
SHADOW_GRASP_TARGET = {
    # Four fingers: curl all three flexion joints to near max (~90 deg = 1.57 rad); spread = 0
    "FFJ0": 1.5, "FFJ1": 1.5, "FFJ2": 1.5, "FFJ3": 0.0,
    "MFJ0": 1.5, "MFJ1": 1.5, "MFJ2": 1.5, "MFJ3": 0.0,
    "RFJ0": 1.5, "RFJ1": 1.5, "RFJ2": 1.5, "RFJ3": 0.0,
    "LFJ0": 1.5, "LFJ1": 1.5, "LFJ2": 1.5, "LFJ3": 0.0, "LFJ4": 0.0,
    # Thumb: THJ0 is distal flex (negative = curl), THJ3 is proximal flex, THJ4 is abduction
    "THJ0": -1.5, "THJ1": 0.0, "THJ2": 0.0, "THJ3": 1.2, "THJ4": 0.8,
    # Wrist neutral
    "WRJ0": 0.0, "WRJ1": 0.0,
}


def load_grasp_yaml(path):
    """Loads the grasp YAML file from the given path and returns the parsed dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_cspace(cspace_dict, dof_names):
    """Converts the cspace_position dict from the YAML into a joint angle array aligned with dof_names order, in radians."""
    q = np.zeros(len(dof_names), dtype=np.float32)
    for i, name in enumerate(dof_names):
        q[i] = float(cspace_dict.get(name, 0.0))
    return q


def set_hand_drive_gains(stage, root_path, stiffness=80.0, damping=10.0, create_if_missing=False):
    """
    Iterates over all joints under the hand USD and sets the angular drive (DriveAPI) stiffness and damping.
    When create_if_missing=True, automatically creates a DriveAPI for joints that have a RevoluteJoint but
    no DriveAPI yet.
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
                if create_if_missing:
                    # Revolute joints can appear as typed prims (GetTypeName == "PhysicsRevoluteJoint")
                    # OR as prims with RevoluteJointAPI applied. Check both.
                    is_revolute = (
                        prim.GetTypeName() == "PhysicsRevoluteJoint"
                        or prim.HasAPI(UsdPhysics.RevoluteJointAPI)
                    )
                    if is_revolute:
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
    Sets the DriveAPI target angle for each joint in the hand USD by joint name.
    target_dict_rad is {joint_name: radians}; internally converted to degrees before writing to USD.
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
    """Applies CollisionAPI to the given USD prim so the physics engine can detect collisions."""
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


def build_shadow_target_array(shadow_dof_names):
    """
    Builds a joint angle array for the Shadow hand grasp target, aligned with shadow_dof_names order.
    Matches each DOF name against SHADOW_GRASP_TARGET keys using substring matching.
    """
    target = np.zeros(len(shadow_dof_names), dtype=np.float32)
    for i, name in enumerate(shadow_dof_names):
        name_upper = name.upper()
        for key, val in SHADOW_GRASP_TARGET.items():
            if key in name_upper or name_upper.endswith(key):
                target[i] = val
                break
    return target


def main():
    """
    Main flow: builds the Isaac Sim scene, loads both the Allegro Hand and Shadow Hand,
    drives both hands to grasp their respective cubes simultaneously via PD control.
    """
    import omni.usd
    from pxr import UsdGeom, Gf, Vt, UsdLux, UsdPhysics, Usd

    stage = omni.usd.get_context().get_stage()

    # ---------- 1) Scene: /World root, ground mesh, ambient light ----------
    if not stage.GetPrimAtPath("/World").IsValid():
        world_xf = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world_xf.GetPrim())

    # Ground: large flat mesh with collision enabled
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

    # Cube for Allegro hand: small cube with rigid body and mass
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

    # Shadow hand root position.
    # After 180° Y rotation: local Y unchanged, local Z negated, local X negated.
    # Knuckle offset from root in world = [+0.033, -0.385, +0.01].
    # At 90° curl, fingertip world position = (knuckle_Y, knuckle_Z - finger_length).
    # Finger length (knuckle to distal tip) = 0.094 m from USD geometry.
    # Target: fingertip Z = 0.02 m at full curl → knuckle_Z = 0.02 + 0.094 = 0.114 → root_Z = 0.104.
    # Target: fingertip Y touches cube near face → cube_near_Y = knuckle_Y = -0.750
    #         → cube_center_Y = -0.750 - 0.05 = -0.800.
    SHADOW_HAND_POSITION = [-0.4, -0.365 + 0.72, 0.14]

    # Cube for Shadow hand. Near face at Y = knuckle_Y = root_Y - 0.385 = -0.750.
    # Cube center Y = -0.750 - 0.05 = -0.800 so at full curl fingertips rest on the near face.
    shadow_cube_path = "/World/ShadowCube"
    SHADOW_CUBE_POSITION = (-0.4, 0, 0.05)
    if not stage.GetPrimAtPath(shadow_cube_path).IsValid():
        shadow_cube_prim = UsdGeom.Cube.Define(stage, shadow_cube_path)
        shadow_cube_prim.CreateSizeAttr(2.0)
        xform_api = UsdGeom.XformCommonAPI(shadow_cube_prim)
        xform_api.SetTranslate(SHADOW_CUBE_POSITION)
        xform_api.SetScale((0.05, 0.05, 0.05))
    shadow_cube_prim = stage.GetPrimAtPath(shadow_cube_path)
    if shadow_cube_prim and shadow_cube_prim.IsValid():
        if not shadow_cube_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(shadow_cube_prim)
        if not shadow_cube_prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(shadow_cube_prim)
            mass_api.CreateMassAttr(0.1)
        _apply_collision_and_preset(shadow_cube_prim)

    # Ambient light
    dome_path = "/World/DomeLight"
    if not stage.GetPrimAtPath(dome_path).IsValid():
        dome = UsdLux.DomeLight.Define(stage, dome_path)
        dome.CreateIntensityAttr(1000.0)

    # ---------- 2) Load Allegro Hand USD ----------
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

    # ---------- 3) Load Shadow Hand USD ----------
    shadow_prim_path = "/World/ShadowHand"
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
        ROBOT_HANDS_DIR = os.path.join(SCRIPT_DIR, "assets", "robots", "robots", "hands")
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

    # ---------- 4) Allegro world pose and drive gains ----------
    HAND_POSITION = [0, 0.13, 0.19]
    HAND_ORIENTATION_QUAT = [-0.65328, -0.65328, 0.2706, -0.2706]  # quaternion W, X, Y, Z
    xform = UsdGeom.Xformable(ref_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*HAND_POSITION))
    q = Gf.Quatd(HAND_ORIENTATION_QUAT[0], Gf.Vec3d(*HAND_ORIENTATION_QUAT[1:]))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(q)
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))

    set_hand_drive_gains(stage, hand_prim_path, stiffness=25.0, damping=5.0)

    # ---------- 5) Shadow world pose and drive gains ----------
    # Shadow hand orientation: 180° rotation around Y-axis.
    # Native orientation: fingers point -Y, curling upward (+Z) with positive joint angles.
    # After 180° Y flip: fingers still point -Y, but now curl DOWNWARD (-Z) with positive angles.
    # Palm still faces -Y (toward the cube). Quaternion: [cos90°, 0, sin90°, 0] = [0, 0, 1, 0] (W,X,Y,Z).
    SHADOW_ORIENTATION_QUAT = [0.0, 0.0, 1.0, 0.0]
    shadow_prim = stage.GetPrimAtPath(shadow_prim_path)
    if shadow_prim and shadow_prim.IsValid():
        shadow_xform = UsdGeom.Xformable(shadow_prim)
        shadow_xform.ClearXformOpOrder()
        shadow_xform.AddTranslateOp().Set(Gf.Vec3d(*SHADOW_HAND_POSITION))
        sq = Gf.Quatd(SHADOW_ORIENTATION_QUAT[0], Gf.Vec3d(*SHADOW_ORIENTATION_QUAT[1:]))
        shadow_xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(sq)
        shadow_xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))

    set_hand_drive_gains(stage, shadow_prim_path, stiffness=25.0, damping=5.0, create_if_missing=True)

    # ---------- 6) Create physics world, register both hands as Articulations ----------
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

    shadow_robot = None
    shadow_art_view = None
    shadow_dof_names = []
    if stage.GetPrimAtPath(shadow_prim_path).IsValid():
        try:
            shadow_robot = world.scene.add(Robot(prim_path=shadow_prim_path, name="shadow_hand"))
        except Exception:
            pass

    world.reset()  # Articulation is only fully initialized after reset

    # Read DOF names after reset
    art_view = getattr(robot, "_articulation_view", None) or getattr(robot, "articulation_view", None)
    dof_names = list(robot.dof_names)
    num_dofs = len(dof_names)

    if shadow_robot is not None:
        shadow_dof_names = list(shadow_robot.dof_names) if (getattr(shadow_robot, "dof_names", None) is not None) else []
        shadow_art_view = getattr(shadow_robot, "_articulation_view", None) or getattr(shadow_robot, "articulation_view", None)

    # ---------- 7) Set world poses after reset ----------
    hand_pos = np.array(HAND_POSITION, dtype=np.float32).reshape(1, 3)
    hand_rot = np.array(HAND_ORIENTATION_QUAT, dtype=np.float32).reshape(1, 4)
    if art_view is not None and hasattr(art_view, "set_world_poses"):
        art_view.set_world_poses(positions=hand_pos, orientations=hand_rot)

    shadow_pos = np.array(SHADOW_HAND_POSITION, dtype=np.float32).reshape(1, 3)
    shadow_rot = np.array(SHADOW_ORIENTATION_QUAT, dtype=np.float32).reshape(1, 4)
    if shadow_art_view is not None and hasattr(shadow_art_view, "set_world_poses"):
        shadow_art_view.set_world_poses(positions=shadow_pos, orientations=shadow_rot)

    # ---------- 8) Load Allegro grasp target from YAML ----------
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

    allegro_initial_q = np.zeros(num_dofs, dtype=np.float32)
    allegro_target_q = parse_cspace(cspace, dof_names)

    # Build Shadow grasp target array aligned with Isaac DOF order
    shadow_initial_q = np.zeros(len(shadow_dof_names), dtype=np.float32)
    shadow_target_q = build_shadow_target_array(shadow_dof_names)

    # ---------- 9) Teleport both hands to initial (open) pose ----------
    if art_view is not None:
        for fn_pos, fn_vel in [
            ("set_joint_positions", "set_joint_velocities"),
            ("set_dof_positions", "set_dof_velocities"),
        ]:
            if hasattr(art_view, fn_pos):
                try:
                    getattr(art_view, fn_pos)(allegro_initial_q.reshape(1, -1))
                    getattr(art_view, fn_vel)(np.zeros((1, num_dofs), dtype=np.float32))
                    break
                except Exception:
                    pass

    if shadow_art_view is not None and len(shadow_dof_names) > 0:
        for fn_pos, fn_vel in [
            ("set_joint_positions", "set_joint_velocities"),
            ("set_dof_positions", "set_dof_velocities"),
        ]:
            if hasattr(shadow_art_view, fn_pos):
                try:
                    getattr(shadow_art_view, fn_pos)(shadow_initial_q.reshape(1, -1))
                    getattr(shadow_art_view, fn_vel)(np.zeros((1, len(shadow_dof_names)), dtype=np.float32))
                    break
                except Exception:
                    pass

    for _ in range(10):
        world.step(render=False)

    # Re-apply world poses after stabilization steps
    if art_view is not None and hasattr(art_view, "set_world_poses"):
        art_view.set_world_poses(positions=hand_pos, orientations=hand_rot)
    if shadow_art_view is not None and hasattr(shadow_art_view, "set_world_poses"):
        shadow_art_view.set_world_poses(positions=shadow_pos, orientations=shadow_rot)

    # ---------- 10) Initialize drive targets ----------
    INTERP_FRAMES = 120

    allegro_drive_target_dict = {name: float(allegro_initial_q[i]) for i, name in enumerate(dof_names)}
    set_drive_targets_via_usd(stage, hand_prim_path, allegro_drive_target_dict)
    if art_view is not None:
        for method in ("set_joint_position_targets", "set_dof_position_targets"):
            if hasattr(art_view, method):
                try:
                    getattr(art_view, method)(allegro_initial_q.reshape(1, -1))
                    break
                except Exception:
                    pass

    shadow_drive_target_dict = {name: float(shadow_initial_q[i]) for i, name in enumerate(shadow_dof_names)}
    set_drive_targets_via_usd(stage, shadow_prim_path, shadow_drive_target_dict)
    if shadow_art_view is not None and len(shadow_dof_names) > 0:
        for method in ("set_joint_position_targets", "set_dof_position_targets"):
            if hasattr(shadow_art_view, method):
                try:
                    getattr(shadow_art_view, method)(shadow_initial_q.reshape(1, -1))
                    break
                except Exception:
                    pass

    # ---------- 11) Main loop ----------
    import omni.timeline
    from omni.kit.app import get_app
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()  # pause on startup; wait for the user to click play
    frame_idx = 0
    loop_start_time = time.perf_counter()
    prev_playing = False
    while simulation_app.is_running():
        is_playing = timeline.is_playing()

        # Detect stop: transition from playing → stopped.
        # Isaac Sim resets Allegro automatically (it has authored USD initial joint positions from
        # its .usd file), but Shadow drives were created programmatically so we reset it explicitly.
        if prev_playing and not is_playing:
            frame_idx = 0
            loop_start_time = time.perf_counter()
            if shadow_art_view is not None and len(shadow_dof_names) > 0:
                for fn_pos, fn_vel in [
                    ("set_joint_positions", "set_joint_velocities"),
                    ("set_dof_positions", "set_dof_velocities"),
                ]:
                    if hasattr(shadow_art_view, fn_pos):
                        try:
                            getattr(shadow_art_view, fn_pos)(shadow_initial_q.reshape(1, -1))
                            getattr(shadow_art_view, fn_vel)(np.zeros((1, len(shadow_dof_names)), dtype=np.float32))
                            break
                        except Exception:
                            pass
                if hasattr(shadow_art_view, "set_world_poses"):
                    shadow_art_view.set_world_poses(positions=shadow_pos, orientations=shadow_rot)

        prev_playing = is_playing
        if not is_playing:
            get_app().update()
            continue
        frame_idx += 1

        if frame_idx <= INTERP_FRAMES:
            t = float(frame_idx) / float(INTERP_FRAMES)

            # Interpolate Allegro drive target
            allegro_current = allegro_initial_q + t * (allegro_target_q - allegro_initial_q)
            allegro_drive_target_dict = {name: float(allegro_current[i]) for i, name in enumerate(dof_names)}
            set_drive_targets_via_usd(stage, hand_prim_path, allegro_drive_target_dict)
            if art_view is not None:
                for method in ("set_joint_position_targets", "set_dof_position_targets"):
                    if hasattr(art_view, method):
                        try:
                            getattr(art_view, method)(allegro_current.reshape(1, -1))
                            break
                        except Exception:
                            pass

            # Interpolate Shadow drive target — identical pattern to Allegro above.
            if shadow_art_view is not None and len(shadow_dof_names) > 0:
                shadow_current = shadow_initial_q + t * (shadow_target_q - shadow_initial_q)
                shadow_drive_target_dict = {name: float(shadow_current[i]) for i, name in enumerate(shadow_dof_names)}
                set_drive_targets_via_usd(stage, shadow_prim_path, shadow_drive_target_dict)
                for method in ("set_joint_position_targets", "set_dof_position_targets"):
                    if hasattr(shadow_art_view, method):
                        try:
                            getattr(shadow_art_view, method)(shadow_current.reshape(1, -1))
                            break
                        except Exception:
                            pass

        world.step(render=True)

        # Force both hands' world poses back each frame to prevent physics drift
        if art_view is not None and hasattr(art_view, "set_world_poses"):
            art_view.set_world_poses(positions=hand_pos, orientations=hand_rot)
        if shadow_art_view is not None and hasattr(shadow_art_view, "set_world_poses"):
            shadow_art_view.set_world_poses(positions=shadow_pos, orientations=shadow_rot)

        # Throttle for the first 120 frames: ~12 seconds total
        if frame_idx <= 120:
            target_elapsed = frame_idx * (12.0 / 120)
            elapsed = time.perf_counter() - loop_start_time
            if elapsed < target_elapsed:
                time.sleep(target_elapsed - elapsed)

    simulation_app.close()


if __name__ == "__main__":
    main()
