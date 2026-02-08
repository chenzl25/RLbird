from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import trange

from .env import SimpleFlappyEnv
from .model import DQN


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s_next: np.ndarray
    done: float


class ReplayBuffer:
    # Randomized replay decorrelates sequential game frames and stabilizes DQN updates.
    def __init__(self, capacity: int = 50_000) -> None:
        self.buf = deque(maxlen=capacity)

    def add(self, t: Transition) -> None:
        self.buf.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)


def choose_action(
    q_net: DQN,
    state: np.ndarray,
    epsilon: float,
    action_dim: int,
    device: torch.device,
) -> int:
    # Epsilon-greedy policy: explore with epsilon, otherwise exploit current Q-net.
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(torch.argmax(q_net(s), dim=1).item())


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    env = SimpleFlappyEnv(seed=args.seed, max_steps=args.max_steps)

    q_net = DQN(env.state_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
    target_net = DQN(env.state_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = Adam(q_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    buffer = ReplayBuffer(capacity=args.buffer_size)

    epsilon = args.eps_start
    eps_decay = (args.eps_start - args.eps_end) / max(1, args.eps_decay_steps)

    best_score = 0
    episode_rewards: list[float] = []
    start_episode = 0

    if args.resume_from:
        # Resume full trainer state when available (model/target/optimizer/epsilon/counters).
        checkpoint = torch.load(args.resume_from, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            q_net.load_state_dict(checkpoint["model_state_dict"])
            if "target_state_dict" in checkpoint:
                target_net.load_state_dict(checkpoint["target_state_dict"])
            else:
                target_net.load_state_dict(q_net.state_dict())
            if "optimizer_state_dict" in checkpoint and not args.no_resume_optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epsilon = float(checkpoint.get("epsilon", epsilon))
            best_score = int(checkpoint.get("best_score", best_score))
            start_episode = int(checkpoint.get("episode", 0))
            print(
                f"Resumed from checkpoint {args.resume_from} "
                f"(episode={start_episode}, epsilon={epsilon:.3f}, best_score={best_score})"
            )
        else:
            # Backward compatibility: resume from plain state_dict model files.
            q_net.load_state_dict(checkpoint)
            target_net.load_state_dict(q_net.state_dict())
            print(f"Loaded model weights from {args.resume_from}")

    episodes_done = 0
    try:
        for ep in trange(args.episodes, desc="Training"):
            global_ep = start_episode + ep + 1
            state = np.array(env.reset(), dtype=np.float32)
            ep_reward = 0.0

            for _ in range(args.max_steps):
                action = choose_action(q_net, state, epsilon, env.action_dim, device)
                out = env.step(action)

                next_state = np.array(out.state, dtype=np.float32)
                buffer.add(
                    Transition(
                        s=state,
                        a=action,
                        r=out.reward,
                        s_next=next_state,
                        done=float(out.done),
                    )
                )
                state = next_state
                ep_reward += out.reward

                if len(buffer) >= args.batch_size:
                    batch = buffer.sample(args.batch_size)
                    s = torch.tensor(np.array([t.s for t in batch]), dtype=torch.float32, device=device)
                    a = torch.tensor([t.a for t in batch], dtype=torch.long, device=device).unsqueeze(1)
                    r = torch.tensor([t.r for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
                    s_next = torch.tensor(np.array([t.s_next for t in batch]), dtype=torch.float32, device=device)
                    done = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)

                    q_values = q_net(s).gather(1, a)

                    with torch.no_grad():
                        # Bellman target: immediate reward + discounted best next-state value.
                        max_next_q = target_net(s_next).max(dim=1, keepdim=True).values
                        target = r + args.gamma * (1.0 - done) * max_next_q

                    loss = loss_fn(q_values, target)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
                    optimizer.step()

                if out.done:
                    best_score = max(best_score, out.info["score"])
                    break

            episode_rewards.append(ep_reward)
            episodes_done = ep + 1
            epsilon = max(args.eps_end, epsilon - eps_decay)

            # Target net lags behind online net to reduce feedback instability.
            if global_ep % args.target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

            if global_ep % args.log_every == 0:
                recent = episode_rewards[-args.log_every :]
                avg_reward = sum(recent) / len(recent)
                print(
                    f"Episode {global_ep:4d} | avg_reward={avg_reward:6.2f} "
                    f"| epsilon={epsilon:5.3f} | best_score={best_score}"
                )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
    finally:
        total_episodes = start_episode + episodes_done
        # Save both formats: lightweight weights for play.py and full checkpoint for resume.
        checkpoint = {
            "model_state_dict": q_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epsilon": epsilon,
            "best_score": best_score,
            "episode": total_episodes,
            "args": vars(args),
        }
        torch.save(checkpoint, args.checkpoint_out)
        torch.save(q_net.state_dict(), args.model_out)
        print(
            f"Saved model to {args.model_out} and checkpoint to {args.checkpoint_out} "
            f"(episodes_done={total_episodes}, best_score={best_score})"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN on a simple Flappy Bird environment")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=10_000)
    p.add_argument("--target-update-interval", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--model-out", type=str, default="dqn_flappy.pt")
    p.add_argument("--checkpoint-out", type=str, default="dqn_flappy_checkpoint.pt")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--no-resume-optimizer", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
