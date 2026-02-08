from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame
import torch

from .env import SimpleFlappyEnv
from .model import DQN


SKY = (149, 212, 255)
PIPE = (69, 168, 61)
BIRD = (245, 191, 79)
TEXT = (24, 35, 52)
GROUND = (238, 220, 172)


def greedy_action(model: DQN, state: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(torch.argmax(model(s), dim=1).item())


def draw_scene(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env: SimpleFlappyEnv,
    mode: str,
    episode: int,
    done: bool,
) -> None:
    width, height = screen.get_size()
    data = env.render_data()

    screen.fill(SKY)

    pipe_x = int(float(data["pipe_x"]) * width)
    pipe_w = max(16, int(float(data["pipe_width"]) * width))
    gap_y = int(float(data["gap_y"]) * height)
    gap_px = int(float(data["gap_size"]) * height)
    gap_top = max(0, gap_y - gap_px // 2)
    gap_bottom = min(height, gap_y + gap_px // 2)

    pygame.draw.rect(screen, PIPE, (pipe_x, 0, pipe_w, gap_top))
    pygame.draw.rect(screen, PIPE, (pipe_x, gap_bottom, pipe_w, height - gap_bottom))

    ground_h = 28
    pygame.draw.rect(screen, GROUND, (0, height - ground_h, width, ground_h))

    bird_x = int(float(data["bird_x"]) * width)
    bird_y = int(float(data["bird_y"]) * height)
    pygame.draw.circle(screen, BIRD, (bird_x, bird_y), 12)

    score = int(data["score"])
    labels = [
        f"mode={mode}",
        f"episode={episode}",
        f"score={score}",
        "SPACE to flap (human), R to reset, ESC to quit",
    ]
    if done:
        labels.append("episode ended")

    y = 8
    for label in labels:
        surface = font.render(label, True, TEXT)
        screen.blit(surface, (8, y))
        y += 24


def run(args: argparse.Namespace) -> None:
    pygame.init()
    pygame.display.set_caption("Flappy RL Tutorial")
    screen = pygame.display.set_mode((args.width, args.height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 22)

    env = SimpleFlappyEnv(seed=args.seed, max_steps=args.max_steps)
    state = np.array(env.reset(), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = None
    if args.mode == "ai":
        model = DQN(env.state_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train first or pass --model-path."
            )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    running = True
    episode = 1
    flap_requested = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = np.array(env.reset(), dtype=np.float32)
                    episode += 1
                elif event.key == pygame.K_SPACE:
                    flap_requested = True

        if args.mode == "human":
            action = 1 if flap_requested else 0
            flap_requested = False
        else:
            assert model is not None
            action = greedy_action(model, state, device)

        out = env.step(action)
        state = np.array(out.state, dtype=np.float32)

        draw_scene(screen, font, env, args.mode, episode, out.done)
        pygame.display.flip()

        if out.done and args.auto_reset:
            pygame.time.delay(args.reset_delay_ms)
            state = np.array(env.reset(), dtype=np.float32)
            episode += 1

        clock.tick(args.fps)

    pygame.quit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play/render the Flappy RL environment with pygame")
    p.add_argument("--mode", choices=["human", "ai"], default="human")
    p.add_argument("--model-path", type=str, default="dqn_flappy.pt")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--reset-delay-ms", type=int, default=600)
    p.add_argument("--auto-reset", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
