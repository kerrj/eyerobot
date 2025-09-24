# Import and re-export all agent classes from separate files
from eye.agents.visual_search_agent import VisualSearchAgent
from eye.agents.eye_robot_agent import EyeRobotAgent
from eye.agents.robot_agent import RobotAgent

# Re-export for backward compatibility
__all__ = [
    'VisualSearchAgent',
    'EyeRobotAgent', 
    'RobotAgent'
]

