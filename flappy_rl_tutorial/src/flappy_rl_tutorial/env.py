from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class StepResult:
    state: list[float]
    reward: float
    done: bool
    info: dict


class SimpleFlappyEnv:
    """A tiny Flappy Bird-like environment for RL learning.

    State features:
    - y: bird vertical position (0..1)
    - vy: bird vertical velocity (scaled)
    - pipe_x: next pipe x position (0..1)
    - gap_y: center of pipe gap (0..1)
    - dist_to_gap: y - gap_y
    - time_alive: normalized by max_steps

    Actions:
    - 0: do nothing
    - 1: flap (negative vertical impulse)
    """

    def __init__(
        self,
        gravity: float = 0.0035,
        flap_impulse: float = -0.03,
        pipe_speed: float = 0.01,
        gap_size: float = 0.28,
        bird_x: float = 0.2,
        pipe_width: float = 0.14,
        max_steps: int = 2000,
        seed: int | None = 0,
    ) -> None:
        self.gravity = gravity
        self.flap_impulse = flap_impulse
        self.pipe_speed = pipe_speed
        self.gap_size = gap_size
        self.bird_x = bird_x
        self.pipe_width = pipe_width
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.reset()

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def action_dim(self) -> int:
        return 2

    def _new_pipe(self) -> tuple[float, float]:
        return 1.0, self.rng.uniform(0.25, 0.75)

    def _get_state(self) -> list[float]:
        dist_to_gap = self.y - self.gap_y
        t_alive = min(self.t / self.max_steps, 1.0)
        return [
            self.y,
            self.vy * 20.0,
            self.pipe_x,
            self.gap_y,
            dist_to_gap,
            t_alive,
        ]

    def reset(self) -> list[float]:
        self.y = 0.5
        self.vy = 0.0
        self.pipe_x, self.gap_y = self._new_pipe()
        self.pipe_passed = False
        self.t = 0
        self.score = 0
        self.done = False
        return self._get_state()

    def render_data(self) -> dict[str, float | int]:
        return {
            "bird_x": self.bird_x,
            "bird_y": self.y,
            "pipe_x": self.pipe_x,
            "pipe_width": self.pipe_width,
            "gap_y": self.gap_y,
            "gap_size": self.gap_size,
            "score": self.score,
            "t": self.t,
        }

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(self._get_state(), 0.0, True, self.render_data())

        self.t += 1

        if action == 1:
            self.vy = self.flap_impulse

        self.vy += self.gravity
        self.y += self.vy
        self.pipe_x -= self.pipe_speed

        reward = 0.01

        half_gap = self.gap_size / 2.0
        inside_gap = self.gap_y - half_gap <= self.y <= self.gap_y + half_gap
        overlaps_pipe = self.pipe_x <= self.bird_x <= self.pipe_x + self.pipe_width
        if overlaps_pipe and not inside_gap:
            self.done = True
            reward = -1.0

        passed_pipe = (self.pipe_x + self.pipe_width) < self.bird_x
        if passed_pipe and not self.pipe_passed and not self.done:
            self.pipe_passed = True
            self.score += 1
            reward = 1.0
            self.pipe_x, self.gap_y = self._new_pipe()
            self.pipe_passed = False

        if self.y < 0.0 or self.y > 1.0:
            self.done = True
            reward = -1.0

        if self.t >= self.max_steps:
            self.done = True

        return StepResult(self._get_state(), reward, self.done, self.render_data())
