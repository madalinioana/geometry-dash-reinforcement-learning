"""
Play Geometry Dash manually with keyboard controls.
Press SPACE or UP to jump.
Press ESC or close window to quit.
"""

import pygame
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import ImpossibleGameEnv


def play_game():
    """Play the game manually with keyboard controls."""

    print("\n" + "="*50)
    print("GEOMETRY DASH - Manual Play")
    print("="*50)
    print("Controls:")
    print("  SPACE / UP Arrow  - Jump")
    print("  ESC              - Quit")
    print("="*50 + "\n")

    env = ImpossibleGameEnv(render_mode="human", max_steps=100000)
    obs, info = env.reset()

    running = True
    total_score = 0
    games_played = 0

    while running:
        action = 0  # Default: no jump

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_UP):
                    action = 1  # Jump

        # Also check for held keys (continuous jump while holding)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            games_played += 1
            total_score += info['score']
            print(f"Game Over! Score: {info['score']} | Average: {total_score/games_played:.0f}")
            obs, info = env.reset()

    env.close()

    print("\n" + "="*50)
    print(f"Games played: {games_played}")
    if games_played > 0:
        print(f"Best score: {total_score}")
        print(f"Average score: {total_score/games_played:.0f}")
    print("="*50)


if __name__ == "__main__":
    play_game()
