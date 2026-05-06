"""TCP -> NavigateToPose proxy.

Runs inside WSL2 alongside Nav2. Listens for one-line goal commands from the
Isaac Sim runner on the Windows host, and dispatches them as Nav2 action goals.

Wire format (one JSON object per line over TCP):
    {"x": 5.0, "y": 18.0, "yaw": 1.5708}    -> send NavigateToPose
    {"cancel": true}                         -> cancel current goal

Why not pubsub a PoseStamped through the Isaac Sim ROS 2 bridge directly?
- Composing PoseStamped messages from Isaac Sim's OmniGraph is awkward.
- Goals are sparse, low-rate commands; a TCP socket is dead-simple and avoids
  any DDS discovery quirks across the Windows/WSL2 NAT boundary.
- Streaming data (odom, tf, cmd_vel) DOES go over DDS — that's what the bridge
  is good at.
"""
from __future__ import annotations

import json
import math
import socketserver
import threading

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


HOST = "0.0.0.0"
PORT = 5005


def yaw_to_quat(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class GoalProxyNode(Node):
    def __init__(self):
        super().__init__("nav2_goal_proxy")
        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._current_goal_handle = None
        self.get_logger().info("waiting for nav2 navigate_to_pose action server...")
        self._client.wait_for_server()
        self.get_logger().info("nav2 action server ready.")

    def send_goal(self, x: float, y: float, yaw: float) -> None:
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        qx, qy, qz, qw = yaw_to_quat(yaw)
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        goal.pose = ps

        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)
        self.get_logger().info(f"sent goal: ({x:.2f}, {y:.2f}, yaw={yaw:.2f})")

    def cancel(self) -> None:
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
            self.get_logger().info("cancel requested")

    def _on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("nav2 rejected goal")
            return
        self._current_goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        result = future.result()
        status = {GoalStatus.STATUS_SUCCEEDED: "succeeded",
                  GoalStatus.STATUS_ABORTED:  "aborted",
                  GoalStatus.STATUS_CANCELED: "canceled"}.get(result.status, str(result.status))
        self.get_logger().info(f"goal result: {status}")
        self._current_goal_handle = None


def make_handler(node: GoalProxyNode):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for raw in self.rfile:
                try:
                    msg = json.loads(raw.decode("utf-8").strip())
                except json.JSONDecodeError as e:
                    node.get_logger().warn(f"bad json: {e}")
                    continue
                if msg.get("cancel"):
                    node.cancel()
                    continue
                try:
                    node.send_goal(float(msg["x"]), float(msg["y"]), float(msg.get("yaw", 0.0)))
                except (KeyError, ValueError) as e:
                    node.get_logger().warn(f"bad goal payload {msg}: {e}")
    return Handler


def main():
    rclpy.init()
    node = GoalProxyNode()

    server = socketserver.ThreadingTCPServer((HOST, PORT), make_handler(node))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    node.get_logger().info(f"goal proxy listening on {HOST}:{PORT}")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
