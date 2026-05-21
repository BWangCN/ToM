"""Isaac Sim ROS 2 bridge wiring for the Split Decision evader.

Runs ONLY on the Windows side, inside the SimulationApp. Builds an OmniGraph
that:
    publishes /tf, /tf_static, /evader/odom        (60 Hz)
    subscribes /evader/cmd_vel -> articulation     (Twist -> diff drive)

The defender stays on the heading-P controller in rule_based.py — only the
evader is exposed to Nav2 for V1, see progress.md / Nav2 design choice D.

Goal injection (PoseStamped to Nav2) does NOT go through this bridge; it's
sent over a TCP socket to nav2_goal_proxy.py running in WSL2. See
runner.py and ros2/nav2_bridge/nav2_goal_proxy.py.

Must be imported AFTER SimulationApp is alive.
"""
from __future__ import annotations

import json
import socket
from typing import Optional


# ROS 2 bridge extension — Isaac Sim 5.x naming. The 4.x name was
# omni.isaac.ros2_bridge; if you're on 4.x change the string here.
_BRIDGE_EXT = "isaacsim.ros2.bridge"


def enable_ros2_bridge() -> None:
    """Idempotently enable the ROS 2 bridge extension."""
    import omni.kit.app
    mgr = omni.kit.app.get_app().get_extension_manager()
    if not mgr.is_extension_enabled(_BRIDGE_EXT):
        mgr.set_extension_enabled_immediate(_BRIDGE_EXT, True)


def build_evader_graph(evader_prim: str = "/World/Evader",
                       odom_topic: str = "/evader/odom",
                       cmd_vel_topic: str = "/evader/cmd_vel",
                       wheel_radius: float = 0.0325,
                       wheel_distance: float = 0.1185) -> None:
    """Wire an OmniGraph that publishes odom/tf and subscribes cmd_vel.

    The graph runs on the simulation tick. We use the standard Isaac Sim
    composite nodes so we don't have to hand-roll Twist→velocity math.
    """
    import omni.graph.core as og

    keys = og.Controller.Keys
    graph_path = "/World/EvaderROS2Graph"

    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnTick",         "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime",    "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context",        "isaacsim.ros2.bridge.ROS2Context"),
                # outbound
                ("ComputeOdom",    "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PublishOdom",    "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("PublishRawTf",   "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ("PublishTf",      "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                # inbound
                ("SubTwist",       "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ("DiffController", "isaacsim.robot.wheeled_robots.DifferentialController"),
                ("ArticulationCtl","isaacsim.core.nodes.IsaacArticulationController"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick",                 "ComputeOdom.inputs:execIn"),
                ("OnTick.outputs:tick",                 "PublishOdom.inputs:execIn"),
                ("OnTick.outputs:tick",                 "PublishRawTf.inputs:execIn"),
                ("OnTick.outputs:tick",                 "PublishTf.inputs:execIn"),
                ("OnTick.outputs:tick",                 "SubTwist.inputs:execIn"),
                ("OnTick.outputs:tick",                 "DiffController.inputs:execIn"),
                ("OnTick.outputs:tick",                 "ArticulationCtl.inputs:execIn"),

                ("Context.outputs:context",             "PublishOdom.inputs:context"),
                ("Context.outputs:context",             "PublishRawTf.inputs:context"),
                ("Context.outputs:context",             "PublishTf.inputs:context"),
                ("Context.outputs:context",             "SubTwist.inputs:context"),

                ("ReadSimTime.outputs:simulationTime",  "PublishOdom.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime",  "PublishRawTf.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime",  "PublishTf.inputs:timeStamp"),

                ("ComputeOdom.outputs:linearVelocity",  "PublishOdom.inputs:linearVelocity"),
                ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
                ("ComputeOdom.outputs:position",        "PublishOdom.inputs:position"),
                ("ComputeOdom.outputs:orientation",     "PublishOdom.inputs:orientation"),
                ("ComputeOdom.outputs:position",        "PublishRawTf.inputs:translation"),
                ("ComputeOdom.outputs:orientation",     "PublishRawTf.inputs:rotation"),

                ("SubTwist.outputs:linearVelocity",     "DiffController.inputs:linearVelocity"),
                ("SubTwist.outputs:angularVelocity",    "DiffController.inputs:angularVelocity"),

                ("DiffController.outputs:velocityCommand",
                                                        "ArticulationCtl.inputs:velocityCommand"),
            ],
            keys.SET_VALUES: [
                ("PublishOdom.inputs:topicName",        odom_topic),
                ("PublishOdom.inputs:odomFrameId",      "odom"),
                ("PublishOdom.inputs:chassisFrameId",   "base_link"),
                ("PublishRawTf.inputs:parentFrameId",   "odom"),
                ("PublishRawTf.inputs:childFrameId",    "base_link"),
                ("ComputeOdom.inputs:chassisPrim",      [evader_prim]),

                ("SubTwist.inputs:topicName",           cmd_vel_topic),

                ("DiffController.inputs:wheelRadius",   wheel_radius),
                ("DiffController.inputs:wheelDistance", wheel_distance),

                ("ArticulationCtl.inputs:targetPrim",   [evader_prim]),
                ("ArticulationCtl.inputs:jointNames",   ["left_wheel_joint", "right_wheel_joint"]),
            ],
        },
    )


# --------------------------------------------------------------------------
# TCP client to nav2_goal_proxy.py running in WSL2.
# --------------------------------------------------------------------------

class GoalProxyClient:
    """Persistent TCP client. Reconnects lazily — Nav2 may come up after Isaac."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5005):
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None

    def _ensure(self) -> Optional[socket.socket]:
        if self._sock is not None:
            return self._sock
        try:
            s = socket.create_connection((self.host, self.port), timeout=0.5)
            s.settimeout(None)
            self._sock = s
            return s
        except OSError as e:
            print(f"[ros2_bridge] goal proxy not reachable at {self.host}:{self.port} ({e}); "
                  "is nav2_goal_proxy.py running in WSL2?")
            return None

    def send_goal(self, x: float, y: float, yaw: float = 0.0) -> bool:
        s = self._ensure()
        if s is None:
            return False
        payload = (json.dumps({"x": float(x), "y": float(y), "yaw": float(yaw)}) + "\n").encode("utf-8")
        try:
            s.sendall(payload)
            return True
        except OSError:
            self._sock = None
            return False

    def cancel(self) -> bool:
        s = self._ensure()
        if s is None:
            return False
        try:
            s.sendall(b'{"cancel": true}\n')
            return True
        except OSError:
            self._sock = None
            return False

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
