"""Bring up Nav2 + map_server + goal_proxy for the Split Decision evader.

Run inside WSL2:
    ros2 launch IsaacToM/ros2/launch/split_decision_nav2.launch.py
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    here = Path(__file__).resolve().parents[1]
    params_file = str(here / "config" / "nav2_params.yaml")
    map_yaml = str(here / "maps" / "empty_arena.yaml")
    proxy_script = str(here / "nav2_bridge" / "nav2_goal_proxy.py")

    nav2_bringup = get_package_share_directory("nav2_bringup")
    bringup_launch = str(Path(nav2_bringup) / "launch" / "bringup_launch.py")

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup_launch),
            launch_arguments={
                "use_sim_time": "true",
                "autostart":    "true",
                "map":          map_yaml,
                "params_file":  params_file,
            }.items(),
        ),
        ExecuteProcess(
            cmd=["python3", proxy_script],
            output="screen",
        ),
    ])
