#!/usr/bin/env python3
"""
Motor controller node for a 2-wheel differential drive robot.
Controls 2 DC motors via an L298N H-bridge on Raspberry Pi 5 GPIO (lgpio).

Modes:
  Hardware  – drives GPIO pins via L298N
  Simulate  – publishes cmd_vel Twist for the Gazebo diff_drive plugin

Subscriptions:
  /cmd_vel (geometry_msgs/Twist)    - velocity commands (hardware mode only)
  /motor_command (std_msgs/String)  - simple text: forward, backward, left, right, stop
  /cliff_scan (sensor_msgs/LaserScan) - cliff sensor (when cliff safety is enabled)

Publishers:
  /cmd_vel (geometry_msgs/Twist)    - velocity output (simulation mode only)

L298N wiring (default GPIO pins, configurable via ROS params):
  Motor A (left):  ENA=12  IN1=23  IN2=24
  Motor B (right): ENB=13  IN3=27  IN4=22
"""

import math
import os
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

try:
    import lgpio
except ImportError:
    lgpio = None


def _open_gpiochip():
    """Open the first usable gpiochip (GANGUBAI_GPIOCHIP env var overrides)."""
    override = os.environ.get('GANGUBAI_GPIOCHIP', '').strip()
    candidates = [int(override)] if override.isdigit() else []
    candidates += sorted(
        int(p.name.removeprefix('gpiochip'))
        for p in Path('/dev').glob('gpiochip*')
        if p.name.removeprefix('gpiochip').isdigit()
    )
    last_error = None
    for chip_index in candidates:
        try:
            return lgpio.gpiochip_open(chip_index), chip_index
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'Unable to open any gpiochip. Tried {candidates}. Last error: {last_error}')


class MotorControllerNode(Node):
    """ROS 2 node that drives two DC motors through an L298N module."""

    def __init__(self):
        super().__init__('motor_controller')

        if lgpio is None:
            self.get_logger().error('lgpio not installed. Forcing simulation mode.')

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter('ena_pin', 12)   # PWM pin for left motor
        self.declare_parameter('in1_pin', 23)
        self.declare_parameter('in2_pin', 24)
        self.declare_parameter('enb_pin', 13)   # PWM pin for right motor
        self.declare_parameter('in3_pin', 27)
        self.declare_parameter('in4_pin', 22)
        self.declare_parameter('pwm_freq', 1000)          # Hz
        self.declare_parameter('max_speed', 100.0)         # duty-cycle %
        self.declare_parameter('default_speed', 70.0)      # duty-cycle % for string commands
        self.declare_parameter('linear_scale', 0.3)        # m/s per 100% speed (for sim Twist)
        self.declare_parameter('angular_scale', 1.0)       # rad/s per 100% speed (for sim Twist)
        self.declare_parameter('cmd_vel_timeout', 0.5)     # seconds without command → stop
        self.declare_parameter('simulate', lgpio is None)
        self.declare_parameter('cliff_topic', 'cliff_scan')
        self.declare_parameter('cliff_edge_distance_m', 0.14)
        self.declare_parameter('cliff_topic_timeout_s', 1.0)
        self.declare_parameter('enable_cliff_safety', self.get_parameter('simulate').value)

        self.ena_pin = self.get_parameter('ena_pin').value
        self.in1_pin = self.get_parameter('in1_pin').value
        self.in2_pin = self.get_parameter('in2_pin').value
        self.enb_pin = self.get_parameter('enb_pin').value
        self.in3_pin = self.get_parameter('in3_pin').value
        self.in4_pin = self.get_parameter('in4_pin').value
        self.pwm_freq = self.get_parameter('pwm_freq').value
        self.max_speed = self.get_parameter('max_speed').value
        self.default_speed = self.get_parameter('default_speed').value
        self.linear_scale = self.get_parameter('linear_scale').value
        self.angular_scale = self.get_parameter('angular_scale').value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value
        self.simulate = self.get_parameter('simulate').value
        self.cliff_edge_distance_m = float(self.get_parameter('cliff_edge_distance_m').value)
        self.cliff_topic_timeout_s = float(self.get_parameter('cliff_topic_timeout_s').value)
        self.enable_cliff_safety = bool(self.get_parameter('enable_cliff_safety').value)

        # ── Hardware or simulation I/O ───────────────────────────────────
        # Hardware: subscribe to cmd_vel and drive GPIO.
        # Simulation: PUBLISH cmd_vel for the Gazebo diff_drive plugin.
        if self.simulate:
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
            mode = 'SIMULATION (Gazebo)'
        else:
            self._setup_gpio()
            self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)
            mode = f'HARDWARE (lgpio chip {self._chip_index})'

        self.create_subscription(String, 'motor_command', self._motor_command_cb, 10)

        self._last_cliff_time = None
        self._last_cliff_range = None
        if self.enable_cliff_safety:
            cliff_topic = str(self.get_parameter('cliff_topic').value)
            self.create_subscription(LaserScan, cliff_topic, self._cliff_scan_cb, qos_profile_sensor_data)

        # ── Safety watchdog: stop motors if no command received ──────────
        self._last_cmd_time = self.get_clock().now()
        self._stopped = True
        self.create_timer(0.1, self._watchdog_cb)

        self.get_logger().info(f'Motor controller started in {mode} mode')

    # ═══════════════════════════════════════════════════════════════════
    # Cliff safety
    # ═══════════════════════════════════════════════════════════════════

    def _cliff_scan_cb(self, msg: LaserScan):
        finite_ranges = [r for r in msg.ranges if math.isfinite(r)]
        self._last_cliff_time = self.get_clock().now()
        self._last_cliff_range = min(finite_ranges) if finite_ranges else None

    def _cliff_blocks_forward(self) -> bool:
        if not self.enable_cliff_safety:
            return False

        if self._last_cliff_time is None or (
            (self.get_clock().now() - self._last_cliff_time).nanoseconds / 1e9 > self.cliff_topic_timeout_s
        ):
            self.get_logger().warn('Cliff safety active but no fresh cliff data; blocking forward motion.')
            return True

        if self._last_cliff_range is not None and self._last_cliff_range >= self.cliff_edge_distance_m:
            self.get_logger().warn('Cliff detected; blocking forward motion.')
            return True

        return False

    # ═══════════════════════════════════════════════════════════════════
    # Motor output
    # ═══════════════════════════════════════════════════════════════════

    def _setup_gpio(self):
        """Claim direction pins as outputs (hardware mode only)."""
        self._chip, self._chip_index = _open_gpiochip()
        for pin in (self.in1_pin, self.in2_pin, self.in3_pin, self.in4_pin, self.ena_pin, self.enb_pin):
            lgpio.gpio_claim_output(self._chip, pin, 0)

    def _set_motors(self, left_speed: float, right_speed: float):
        """Set motor speeds in -100.0 (full reverse) … +100.0 (full forward)."""
        left_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        right_speed = max(-self.max_speed, min(self.max_speed, right_speed))

        if self.simulate:
            self._publish_twist(left_speed, right_speed)
        else:
            self._apply_motor(self.in1_pin, self.in2_pin, self.ena_pin, left_speed)
            self._apply_motor(self.in3_pin, self.in4_pin, self.enb_pin, right_speed)

        self._stopped = (left_speed == 0.0 and right_speed == 0.0)

    def _publish_twist(self, left_speed: float, right_speed: float):
        """Convert left/right motor percentages to a Twist for Gazebo."""
        twist = Twist()
        twist.linear.x = (left_speed + right_speed) / 2.0 / self.max_speed * self.linear_scale
        twist.angular.z = (right_speed - left_speed) / 2.0 / self.max_speed * self.angular_scale
        self.cmd_vel_pub.publish(twist)

    def _apply_motor(self, in_a: int, in_b: int, enable_pin: int, speed: float):
        """Drive one motor at the given signed speed (-100…+100)."""
        lgpio.gpio_write(self._chip, in_a, 1 if speed > 0 else 0)
        lgpio.gpio_write(self._chip, in_b, 1 if speed < 0 else 0)
        lgpio.tx_pwm(self._chip, enable_pin, self.pwm_freq, min(100.0, abs(speed)))

    def _stop_motors(self):
        self._set_motors(0.0, 0.0)

    # ═══════════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════════

    def _cmd_vel_cb(self, msg: Twist):
        """Convert Twist to differential drive (hardware mode only)."""
        self._last_cmd_time = self.get_clock().now()
        linear = max(-1.0, min(1.0, msg.linear.x))
        angular = max(-1.0, min(1.0, msg.angular.z))
        self._set_motors((linear - angular) * self.max_speed, (linear + angular) * self.max_speed)

    def _motor_command_cb(self, msg: String):
        """Handle simple text commands (works in both modes)."""
        command = msg.data.strip().lower()
        speed = self.default_speed
        speeds = {
            'forward': (speed, speed),
            'backward': (-speed, -speed),
            'left': (-speed, speed),
            'right': (speed, -speed),
            'stop': (0.0, 0.0),
        }
        if command not in speeds:
            self.get_logger().warn(f'Unknown command: "{command}"')
            return

        if command == 'forward' and self._cliff_blocks_forward():
            command = 'stop'

        self._set_motors(*speeds[command])
        self._last_cmd_time = self.get_clock().now()

    def _watchdog_cb(self):
        """Stop motors if no command received within timeout."""
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9
        if elapsed > self.cmd_vel_timeout and not self._stopped:
            self.get_logger().info('Timeout — stopping motors')
            self._stop_motors()

    def destroy_node(self):
        self._stop_motors()
        if not self.simulate:
            for pin in (self.ena_pin, self.enb_pin):
                lgpio.tx_pwm(self._chip, pin, 0, 0.0)
            lgpio.gpiochip_close(self._chip)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
