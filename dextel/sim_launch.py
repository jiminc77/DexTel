import os
import sys
import logging
import numpy as np

sys.path = [p for p in sys.path if "ros/jazzy" not in p]

import isaacsim
from omni.isaac.kit import SimulationApp

LAUNCH_CONFIG = {
    "headless": False,
    "extensions": [
        "isaacsim.ros2.bridge",
        "isaacsim.core.nodes", 
        "omni.graph.nodes",
    ]
}

simulation_app = SimulationApp(LAUNCH_CONFIG)

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.core.nodes")

simulation_app.update()

import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.robots import Robot
from pxr import Usd, UsdPhysics, Sdf

def create_ros2_bridge_graph():
    """Creates the OmniGraph for ROS 2 communication."""
    print("[DexTel] Creating ROS 2 Bridge Action Graph...")
    
    # Check ROS Domain ID
    domain_id = os.environ.get("ROS_DOMAIN_ID")
    print(f"[DexTel] ROS_DOMAIN_ID: {domain_id}")

    keys = og.Controller.Keys
    
    ros_bridge_ns = "isaacsim.ros2.bridge" 
    core_nodes_ns = "isaacsim.core.nodes"
    
    try:
        og.Controller.edit(
            {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ROS2Context", f"{ros_bridge_ns}.ROS2Context"),
                    ("ROS2SubscribeJointState", f"{ros_bridge_ns}.ROS2SubscribeJointState"),
                    ("ArticulationController", f"{core_nodes_ns}.IsaacArticulationController"),
                ],
                keys.CONNECT: [
                    # Joint States -> Articulation
                    ("OnPlaybackTick.outputs:tick", "ROS2SubscribeJointState.inputs:execIn"),
                    ("ROS2Context.outputs:context", "ROS2SubscribeJointState.inputs:context"),
                    ("ROS2SubscribeJointState.outputs:execOut", "ArticulationController.inputs:execIn"),
                    ("ROS2SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("ROS2SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ],
                    keys.SET_VALUES: [
                    # Joint Control Parameters
                    ("ROS2SubscribeJointState.inputs:topicName", "/target_joint_states"),
                    ("ArticulationController.inputs:robotPath", "/World/ur3e"),
                ]
            }
        )
        
        
        # Manually set the targetPrim relationship
        try:
             import omni.usd
             from pxr import Usd, Sdf
             stage = omni.usd.get_context().get_stage()
             prim_path = "/ActionGraph/ArticulationController"
             controller_prim = stage.GetPrimAtPath(prim_path)
             
             if controller_prim.IsValid():
                 print(f"[DexTel] Found ArticulationController at {prim_path}", flush=True)
                 
                 # Create/Get the relationship
                 rel = controller_prim.GetRelationship("inputs:targetPrim")
                 if not rel:
                     rel = controller_prim.CreateRelationship("inputs:targetPrim")
                 
                 # Set the target
                 target_path = Sdf.Path("/World/ur3e")
                 rel.SetTargets([target_path])
                 
                 # Verify
                 current_targets = rel.GetTargets()
                 print(f"[DexTel] Set ArticulationController targetPrim to: {current_targets}", flush=True)
             else:
                     print(f"[DexTel] [ERROR] ArticulationController prim not found at {prim_path}!", flush=True)


                 
        except Exception as rel_err:
             print(f"[DexTel] Error setting targetPrim relationship: {rel_err}", flush=True)
             import traceback
             traceback.print_exc()

    except Exception as e:
        print(f"[DexTel] Error creating ROS 2 Bridge: {e}", flush=True)

        # raise e # Suppress crash to allow debug

def load_scene():
    """Loads the USD stage for the simulation."""
    # Resolve path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    usd_path = os.path.join(current_dir, "assets/ur3e_hande.usd")
    
    if os.path.exists(usd_path):
        print(f"[DexTel] Loading User Stage: {usd_path}")
        open_stage(usd_path)
        return True
    else:
        print(f"[DexTel] [WARN] Asset not found at: {usd_path}")
        print("[DexTel] Starting with empty stage.")
        return False

def main():
    # 1. Load Scene FIRST (This resets the stage)
    scene_loaded = load_scene()
    
    simulation_app.update()
    
    # 3. Initialize World (must be done after stage load)
    world = World(stage_units_in_meters=1.0)
    
    # Initialize World
    if scene_loaded:
        try:
            # Wrap the robot for easier access/manipulation if needed later
            robot = Robot(prim_path="/World/ur3e", name="ur3e")
            world.scene.add(robot)
        except Exception as e:
            print(f"[DexTel] Robot wrapper warning: {e}")

    # 4. Create ROS 2 Bridge (after World init is safe, but technically independent)
    try:
        create_ros2_bridge_graph()
    except Exception:
        print("[DexTel] Bridge creation failed. Running without bridge.")

    # 5. Reset World
    world.reset()
    
    # DEBUG: Check joints in ur3e
    try:
        print(f"[DexTel] UR3e Articulation Initialized. Joints: {robot.dof_names}")
        
        # Configure Gripper Drives (Stiffness/Damping)
        # Assuming last two joints are Slider_1 and Slider_2
        # We need to set them to be position controlled (high stiffness)
        # This requires using the Articulation API from the robot object
        
        # --- Robust USD DriveAPI Configuration ---
        # Instead of runtime gains (which can be flaky), we set the Drive properties directly on the USD Prims.
        
        stage = omni.usd.get_context().get_stage()
        robot_prim_path = "/World/ur3e"
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        
        if robot_prim.IsValid():
            print(f"[DexTel] Scanning for Gripper Joints in {robot_prim_path} to apply DriveAPI...")
            
            gripper_keywords = ["Slider", "finger", "drive", "hand"]
            exclude_keywords = ["shoulder", "elbow", "wrist"]
            
            count = 0
            for prim in Usd.PrimRange(robot_prim):
                if prim.IsA(UsdPhysics.Joint):
                    name = prim.GetName()
                    
                    is_gripper = any(k in name for k in gripper_keywords) and \
                                 not any(k in name for k in exclude_keywords)
                                 
                    if is_gripper:
                        print(f"[DexTel] Found Gripper Joint Prim: {name}")
                        
                        # Determine Drive Type: Prismatic -> linear, Revolute -> angular
                        # Default to angular if unsure, but checking type is better
                        drive_type = "angular"
                        if prim.IsA(UsdPhysics.PrismaticJoint):
                            drive_type = "linear"
                            
                        # Apply Drive API
                        # The API schema is applied to the prim with a specific instance name (drive_type)
                        drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_type)
                        
                        # Set Properties for Position Control (Stiff)
                        # We use a very high stiffness to ensure it holds position
                        drive_api.CreateStiffnessAttr(1.0e5)
                        drive_api.CreateDampingAttr(1.0e4)
                        
                        print(f"    -> Applied {drive_type} DriveAPI: Stiffness=1.0e5, Damping=1.0e4")
                        count += 1
            
            if count == 0:
                print("[DexTel] [WARN] No gripper joints found to configure via USD.")
            else:
                print(f"[DexTel] Successfully configured {count} gripper joints via USD DriveAPI.")
        else:
            print(f"[DexTel] [WARN] Robot prim not found at {robot_prim_path}")

            
    except Exception as e:
        print(f"[DexTel] Could not configure robot/gripper: {e}")
        import traceback
        traceback.print_exc()
        
    print("[DexTel] Simulation Ready. Press PLAY in the Viewport or wait...")
    
    try:
        while simulation_app.is_running():
            world.step(render=True)
            
    except KeyboardInterrupt:
        print("[DexTel] Stopping simulation...")
        
    simulation_app.close()

if __name__ == "__main__":
    main()
