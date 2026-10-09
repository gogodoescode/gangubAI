"""Send movement and wander commands to the ROS 2 robot stack.

Shells out to the `ros2` CLI / system python so the voice app does not need
to spin up its own rclpy node.
"""

import re
import subprocess

DIRECTIONS = ("forward", "backward", "left", "right", "360", "stop")
SPIN_DURATION_S = 10.0

# Keeps re-publishing the command so the motor controller watchdog
# (cmd_vel_timeout ~0.5s) does not stop the robot early, then sends stop.
_MOVE_SCRIPT = """
import sys, time
import rclpy
from std_msgs.msg import String

direction, duration_s = sys.argv[1], float(sys.argv[2])
rclpy.init()
node = rclpy.create_node('direct_motor_publisher')
pub = node.create_publisher(String, '/motor_command', 10)
deadline = time.time() + duration_s
try:
    while time.time() < deadline:
        pub.publish(String(data=direction))
        time.sleep(0.1)
finally:
    pub.publish(String(data='stop'))
    rclpy.shutdown()
"""


def _has_subscribers(topic: str) -> bool:
    try:
        result = subprocess.run(
            ["ros2", "topic", "info", topic],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except Exception:
        return False
    match = re.search(r"Subscription count:\s*(\d+)", result.stdout)
    return result.returncode == 0 and bool(match) and int(match.group(1)) > 0


def _publish_once(topic: str, data: str) -> None:
    subprocess.Popen(
        ["ros2", "topic", "pub", "--once", topic, "std_msgs/msg/String", f"{{data: '{data}'}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def move(direction: str, duration: float | None = None) -> str:
    """Drive in *direction* for *duration* seconds, then auto-stop.

    Default duration is 2s, or a full spin for "360".
    """
    direction = direction.strip().lower()
    if direction not in DIRECTIONS:
        return f"Unknown direction '{direction}'. Valid options: {', '.join(DIRECTIONS)}."

    if not _has_subscribers("/motor_command"):
        return (
            "Error: No subscribers on /motor_command. Start ROS first: "
            "ros2 launch gangubai_control motor_control.launch.py"
        )

    if direction == "stop":
        _publish_once("/motor_command", "stop")
        return "Robot stopped."

    if duration is None:
        duration = SPIN_DURATION_S if direction == "360" else 2.0
    if direction == "360":
        direction = "right"

    duration = max(0.2, float(duration))
    subprocess.Popen(
        ["/usr/bin/python3", "-c", _MOVE_SCRIPT, direction, str(duration)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return f"Robot moving {direction} for {duration:.1f}s then auto-stopping."


def set_wander(action: str, check_subscribers: bool = True) -> str:
    """Start or stop autonomous wander mode."""
    action = action.strip().lower()
    if action not in ("start", "stop"):
        return f"Invalid wander action '{action}'. Use start or stop."

    if check_subscribers and not _has_subscribers("/wander_mode"):
        return "Error: No subscribers on /wander_mode. Start the ROS launch with the wander controller first."

    try:
        _publish_once("/wander_mode", action)
    except FileNotFoundError:
        return "Error: `ros2` command not found. Source ROS 2 first (source /opt/ros/humble/setup.bash)."

    if action == "start":
        return "Wander mode enabled. Robot will move autonomously until stopped."
    return "Wander mode stopped. Robot returned to idle."
