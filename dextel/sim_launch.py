import os
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot

def create_ros2_bridge():
    print("Creating ROS 2 Bridge Action Graph...")
    keys = og.Controller.Keys
    
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2Context", "omni.isaac.ros2_bridge.ROS2Context"),
                ("ROS2SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                
                ("ROS2SubscribeGripper", "omni.isaac.ros2_bridge.ROS2SubscribeFloat64"),
                ("GripperController", "omni.isaac.core_nodes.IsaacArticulationController"),
                ("MakeArray", "omni.graph.nodes.MakeArray"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ROS2SubscribeJointState.inputs:execIn"),
                ("ROS2Context.outputs:context", "ROS2SubscribeJointState.inputs:context"),
                ("ROS2SubscribeJointState.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("ROS2SubscribeJointState.outputs:position", "ArticulationController.inputs:position_command"),
                ("ROS2SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                
                ("OnPlaybackTick.outputs:tick", "ROS2SubscribeGripper.inputs:execIn"),
                ("ROS2Context.outputs:context", "ROS2SubscribeGripper.inputs:context"),
                ("ROS2SubscribeGripper.outputs:execOut", "GripperController.inputs:execIn"),
                ("ROS2SubscribeGripper.outputs:data", "GripperController.inputs:position_command"), 
                ("MakeArray.outputs:array", "GripperController.inputs:jointNames"),
            ],
            keys.SET_VALUES: [
                ("ROS2SubscribeJointState.inputs:topicName", "/target_joint_states"),
                ("ArticulationController.inputs:targetPrim", "/World/ur3e"), 
                ("ArticulationController.inputs:usePath", True),
                
                ("ROS2SubscribeGripper.inputs:topicName", "/gripper_command"),
                ("GripperController.inputs:targetPrim", "/World/robotiq_hande"),
                ("GripperController.inputs:usePath", True),
                ("MakeArray.inputs:input0", "Slider_1"),
                ("MakeArray.inputs:input1", "Slider_2"),
                ("MakeArray.inputs:arraySize", 2),
            ]
        }
    )

def setup_scene(world):
    # 1. Load UR3e
    ur3e_usd_path = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"
    add_reference_to_stage(usd_path=ur3e_usd_path, prim_path="/World/ur3e")

    # 2. Load Hand-E (Assuming path or using a placeholder)
    # Note: Replace with actual path if you have it, or search Nucleus
    hande_usd_path = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/Robotiq/HandE/hande.usd" # check path
    # If not found, user might need to adjust.
    # add_reference_to_stage(usd_path=hande_usd_path, prim_path="/World/robotiq_hande")
    
    # 3. Fix Robot Base to World
    # We create a Fixed Joint between World and base_link
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdPhysics, UsdShade, Gf
    stage = get_current_stage()
    
    # Enable Physics on World if needed? Usually default.
    
    # 4. Attach Gripper to Tool0 (Fixed Joint)
    # Joint Prim Path: /World/robotiq_hande/root_joint (example)
    # We would set Body0 = /World/ur3e/tool0, Body1 = /World/robotiq_hande/base_link
    
    # 5. Configure Joint Drives (Stiffness/Damping)
    # This iterates all joints and sets them to position control with high stiffness
    # (Code omitted for brevity in chat, but would go here)
    pass

def main():
    world = World(stage_units_in_meters=1.0)
    
    # --- OPTION 1: Load User's Manual Stage (Recommended) ---
    # Replace this path with the full path to your saved .usd file
    # e.g., "/home/jimin/Documents/my_test_stage.usd"
    user_stage_path = "/path/to/your/saved_file.usd" 
    
    if os.path.exists(user_stage_path):
        print(f"[INFO] Opening User Stage: {user_stage_path}")
        from omni.isaac.core.utils.stage import open_stage
        open_stage(user_stage_path)
    else:
        print(f"[WARN] File not found: {user_stage_path}")
        print("Please edit 'dextel/sim_launch.py' and set 'user_stage_path' to your .usd file.")
        
    # --- OPTION 2: Automated Setup (Commented Out) ---
    # setup_scene(world) 
    
    # Initialize World (needed to register the robot for simulation steps)
    # If your USD already has the robot at /World/ur3e, we just wrap it.
    try:
        robot = Robot(prim_path="/World/ur3e", name="ur3e")
        world.scene.add(robot)
    except Exception as e:
        print(f"Robot wrapper init warning (ignore if stage not loaded): {e}")

    # Always create the bridge
    create_ros2_bridge()
    
    world.reset()
    print("Simulation Ready. Press PLAY in the Viewport or wait...")
    
    while simulation_app.is_running():
        world.step(render=True)
        
    simulation_app.close()

if __name__ == "__main__":
    main()
