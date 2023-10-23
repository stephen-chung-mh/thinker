# Core Library
import logging

# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="cSokoban-v0", 
    entry_point="gym_csokoban.envs:SokobanEnv",
    kwargs={"difficulty": "unfiltered"},
)
register(
    id="cSokoban-medium-v0",
    entry_point="gym_csokoban.envs:SokobanEnv",
    kwargs={"difficulty": "medium"},
)
register(
    id="cSokoban-hard-v0",
    entry_point="gym_csokoban.envs:SokobanEnv",
    kwargs={"difficulty": "hard"},
)
register(
    id="cSokoban-test-v0",
    entry_point="gym_csokoban.envs:SokobanEnv",
    kwargs={"difficulty": "test"},
)