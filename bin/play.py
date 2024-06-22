import pygame
from src.env import rgb_env
from gym.utils.play import play
from src.constants import ACTION

rgb_env.reset()

mappings = {
    (pygame.K_LEFT,): ACTION.LEFT,
    (pygame.K_RIGHT,): ACTION.RIGHT
}

play(rgb_env, keys_to_action=mappings)
