"""LLM tools that move the robot through ROS 2."""

from typing import Literal

from langchain_core.tools import tool

from ai import robot


@tool
def move_robot(
    direction: Literal["forward", "backward", "left", "right", "360", "stop"],
    duration: float | None = None,
) -> str:
    """Move the Gangubai robot in a given direction.

    Use this tool when the user asks the robot to move, drive, turn, spin,
    go forward/backward, rotate, or do a 360.

    Args:
        direction: The movement direction. One of:
            "forward"  – drive straight ahead
            "backward" – drive in reverse
            "left"     – turn/spin left
            "right"    – turn/spin right
            "360"      – do a full 360° clockwise spin
            "stop"     – immediately stop all motors
        duration: How long to move in seconds. Leave empty for the default
                  (2 seconds, or a full spin for "360"). Ignored for "stop".

    Returns:
        A status message confirming the action.
    """
    return robot.move(direction, duration)


@tool
def set_wander_mode(action: Literal["start", "stop"]) -> str:
    """Start or stop autonomous wander behavior.

    Use this tool when the user asks the robot to wander/explore autonomously,
    or to stop wandering and return to idle.

    Args:
        action: "start" to enter wander mode, "stop" to return to idle.

    Returns:
        A status message confirming the mode change.
    """
    return robot.set_wander(action)
