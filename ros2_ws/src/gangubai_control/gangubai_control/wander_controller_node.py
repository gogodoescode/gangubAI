#!/usr/bin/env python3
"""Autonomous wander behavior controller for GangubAI.

This node runs a simple state machine and publishes string commands to
`/motor_command`, so it works in both simulation and hardware modes through
existing `motor_controller_node` logic.

States:
- idle: no autonomous motion
- wander: random movement with periodic command refresh
- edge_recovery: emergency back-up and turn when cliff edge is detected

Inputs:
- /wander_mode (std_msgs/String): "start" or "stop"
- /cliff_scan (sensor_msgs/LaserScan): cliff safety sensor

Outputs:
- /motor_command (std_msgs/String): forward/backward/left/right/stop
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class WanderControllerNode(Node):
    """State-machine controller for autonomous wandering."""

    @staticmethod
    def _as_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def __init__(self) -> None:
        super().__init__("wander_controller")

        # Topics
        self.declare_parameter("wander_topic", "wander_mode")
        self.declare_parameter("motor_command_topic", "motor_command")
        self.declare_parameter("cliff_topic", "cliff_scan")

        # Loop timing and command refresh
        self.declare_parameter("tick_hz", 10.0)
        self.declare_parameter("command_refresh_s", 0.2)

        # Cliff safety
        self.declare_parameter("require_cliff_data", False)
        self.declare_parameter("cliff_topic_timeout_s", 1.0)
        self.declare_parameter("cliff_edge_distance_m", 0.14)
        self.declare_parameter("cliff_edge_margin_m", 0.005)

        # Wander behavior
        self.declare_parameter("forward_bias", 0.7)
        self.declare_parameter("forward_min_s", 0.8)
        self.declare_parameter("forward_max_s", 2.5)
        self.declare_parameter("turn_min_s", 0.5)
        self.declare_parameter("turn_max_s", 1.4)
        self.declare_parameter("reverse_min_s", 0.7)
        self.declare_parameter("reverse_max_s", 1.2)

        # Edge recovery behavior
        self.declare_parameter("edge_recovery_back_s", 1.0)
        self.declare_parameter("edge_recovery_turn_s", 1.0)

        # Read parameters
        self.wander_topic = str(self.get_parameter("wander_topic").value)
        self.motor_command_topic = str(self.get_parameter("motor_command_topic").value)
        self.cliff_topic = str(self.get_parameter("cliff_topic").value)

        self.tick_hz = float(self.get_parameter("tick_hz").value)
        self.command_refresh_s = float(self.get_parameter("command_refresh_s").value)

        self.require_cliff_data = self._as_bool(
            self.get_parameter("require_cliff_data").value
        )
        self.cliff_topic_timeout_s = float(self.get_parameter("cliff_topic_timeout_s").value)
        self.cliff_edge_distance_m = float(self.get_parameter("cliff_edge_distance_m").value)
        self.cliff_edge_margin_m = float(self.get_parameter("cliff_edge_margin_m").value)

        self.forward_bias = float(self.get_parameter("forward_bias").value)
        self.forward_min_s = float(self.get_parameter("forward_min_s").value)
        self.forward_max_s = float(self.get_parameter("forward_max_s").value)
        self.turn_min_s = float(self.get_parameter("turn_min_s").value)
        self.turn_max_s = float(self.get_parameter("turn_max_s").value)
        self.reverse_min_s = float(self.get_parameter("reverse_min_s").value)
        self.reverse_max_s = float(self.get_parameter("reverse_max_s").value)

        self.edge_recovery_back_s = float(self.get_parameter("edge_recovery_back_s").value)
        self.edge_recovery_turn_s = float(self.get_parameter("edge_recovery_turn_s").value)

        # Publishers/subscribers
        self.motor_pub = self.create_publisher(String, self.motor_command_topic, 10)
        self.create_subscription(String, self.wander_topic, self._wander_mode_cb, 10)
        if self.require_cliff_data:
            self.create_subscription(
                LaserScan, self.cliff_topic, self._cliff_scan_cb, qos_profile_sensor_data
            )

        # Internal state
        self._mode = "idle"  # idle | wander | edge_recovery
        self._current_action = "stop"
        self._action_end_time = 0.0
        self._last_action_sent = 0.0

        self._last_cliff_time = 0.0
        self._last_cliff_range: Optional[float] = None
        self._last_cliff_min: Optional[float] = None
        self._last_cliff_max: Optional[float] = None

        self._recovery_phase = ""
        self._recovery_turn_direction = "left"
        self._recovery_phase_end = 0.0

        period_s = max(0.02, 1.0 / max(0.1, self.tick_hz))
        self._timer = self.create_timer(period_s, self._tick)

        self.get_logger().info(
            "Wander controller ready: mode=idle, "
            f"wander_topic=/{self.wander_topic}, motor_topic=/{self.motor_command_topic}"
        )
        if self.require_cliff_data:
            self.get_logger().info(f"Cliff safety enabled on /{self.cliff_topic}.")
        else:
            self.get_logger().warn("Cliff safety disabled (require_cliff_data=false).")
        self._publish_motor("stop", force=True)

    def _publish_motor(self, action: str, force: bool = False) -> None:
        now = time.monotonic()
        if (
            force
            or action != self._current_action
            or (now - self._last_action_sent) >= self.command_refresh_s
        ):
            self.motor_pub.publish(String(data=action))
            self._current_action = action
            self._last_action_sent = now

    def _enter_idle(self, reason: str) -> None:
        prev_mode = self._mode
        self._mode = "idle"
        self._recovery_phase = ""
        self._action_end_time = 0.0
        self._publish_motor("stop", force=True)

        if prev_mode != "idle":
            self.get_logger().info(f"Wander -> IDLE ({reason})")

    def _enter_wander(self, reason: str) -> None:
        self._mode = "wander"
        self._recovery_phase = ""
        self._schedule_next_action(time.monotonic())
        self.get_logger().info(f"Wander -> WANDER ({reason})")

    def _start_edge_recovery(self, reason: str) -> None:
        now = time.monotonic()
        self._mode = "edge_recovery"
        self._recovery_phase = "backward"
        self._recovery_turn_direction = random.choice(("left", "right"))
        self._recovery_phase_end = now + self.edge_recovery_back_s

        self._publish_motor("backward", force=True)
        self.get_logger().warn(f"Wander -> EDGE_RECOVERY ({reason})")

    def _schedule_next_action(self, now: float) -> None:
        roll = random.random()
        if roll < self.forward_bias:
            action = "forward"
            duration = random.uniform(self.forward_min_s, self.forward_max_s)
        elif roll < (self.forward_bias + 0.2):
            action = random.choice(("left", "right"))
            duration = random.uniform(self.turn_min_s, self.turn_max_s)
        else:
            action = "backward"
            duration = random.uniform(self.reverse_min_s, self.reverse_max_s)

        self._action_end_time = now + duration
        self._publish_motor(action, force=True)

    def _has_fresh_cliff(self, now: float) -> bool:
        return (now - self._last_cliff_time) <= self.cliff_topic_timeout_s

    def _is_edge_detected(self) -> bool:
        if self._last_cliff_range is None:
            return False

        if self._last_cliff_range >= self.cliff_edge_distance_m:
            return True

        if self._last_cliff_max is not None:
            if self._last_cliff_range >= (self._last_cliff_max - self.cliff_edge_margin_m):
                return True

        return False

    def _has_invalid_cliff_reading(self) -> bool:
        if self._last_cliff_range is None:
            return False

        if not math.isfinite(self._last_cliff_range):
            return True

        if self._last_cliff_range < 0.0:
            return True

        if self._last_cliff_min is not None and self._last_cliff_range < self._last_cliff_min:
            return True

        return False

    def _run_recovery_step(self, now: float) -> None:
        if self._recovery_phase == "backward":
            if now >= self._recovery_phase_end:
                self._recovery_phase = "turn"
                self._recovery_phase_end = now + self.edge_recovery_turn_s
                self._publish_motor(self._recovery_turn_direction, force=True)
            else:
                self._publish_motor("backward")
            return

        if self._recovery_phase == "turn":
            if now >= self._recovery_phase_end:
                self._mode = "wander"
                self._schedule_next_action(now)
                self.get_logger().info("Edge recovery complete; resuming wander.")
            else:
                self._publish_motor(self._recovery_turn_direction)
            return

        # Recovery phase was not initialized correctly.
        self._start_edge_recovery("recovery-reset")

    def _tick(self) -> None:
        now = time.monotonic()

        if self._mode == "idle":
            self._publish_motor("stop")
            return

        if self.require_cliff_data and not self._has_fresh_cliff(now):
            self._enter_idle("no-cliff-sensor-data")
            return

        if self.require_cliff_data and self._has_invalid_cliff_reading():
            self.get_logger().warn(
                f"Invalid cliff reading ({self._last_cliff_range}); entering idle for safety."
            )
            self._enter_idle("invalid-cliff-reading")
            return

        if self._is_edge_detected() and self._mode != "edge_recovery":
            self._start_edge_recovery("edge-detected")

        if self._mode == "edge_recovery":
            self._run_recovery_step(now)
            return

        if self._mode == "wander":
            if now >= self._action_end_time:
                self._schedule_next_action(now)
            self._publish_motor(self._current_action)

    def _wander_mode_cb(self, msg: String) -> None:
        command = msg.data.strip().lower()

        if command in ("start", "wander", "on"):
            if self._mode == "wander":
                self.get_logger().info("Wander already active; ignoring duplicate start.")
                return
            self._enter_wander("start-command")
            return

        if command in ("stop", "idle", "off"):
            # Idempotent stop: safe to send repeatedly.
            self._enter_idle("stop-command")
            return

        self.get_logger().warn(f"Unknown wander command: '{command}'")

    def _cliff_scan_cb(self, msg: LaserScan) -> None:
        if not msg.ranges:
            return

        finite_ranges = [r for r in msg.ranges if math.isfinite(r)]
        self._last_cliff_time = time.monotonic()
        if not finite_ranges:
            self._last_cliff_range = float("nan")
            self._last_cliff_min = None
            self._last_cliff_max = None
            return

        self._last_cliff_range = float(min(finite_ranges))
        self._last_cliff_min = float(msg.range_min) if msg.range_min > 0.0 else None
        self._last_cliff_max = float(msg.range_max) if msg.range_max > 0.0 else None

    def destroy_node(self) -> bool:
        try:
            self._publish_motor("stop", force=True)
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WanderControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
