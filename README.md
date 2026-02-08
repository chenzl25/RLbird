# Flappy RL Tutorial (uv + PyTorch)

This is a hands-on tutorial to learn reinforcement learning (RL) by training a small DQN agent on a simple Flappy Bird-style environment.

## 1) What you will learn

- What RL is: agent, environment, state, action, reward
- How Q-learning works with a neural network (DQN)
- Why exploration (epsilon-greedy) and replay buffer matter
- How to train and evaluate a basic RL policy with PyTorch

## 2) Quick start with uv

From this folder:

```bash
cd /Users/dylan/Desktop/playground/RL
uv sync
```

Run quick training:

```bash
uv run python -m flappy_rl_tutorial.train --episodes 400 --log-every 20
```

You should see logs like:

- average reward increasing over time
- best score occasionally improving
- epsilon decreasing from 1.0 to 0.05

The model is saved to `dqn_flappy.pt`.

Run recommended long training (better config):

```bash
uv run python -m flappy_rl_tutorial.train \
  --episodes 15000 \
  --max-steps 3000 \
  --batch-size 128 \
  --lr 0.0007 \
  --eps-start 1.0 \
  --eps-end 0.02 \
  --eps-decay-steps 3000 \
  --target-update-interval 20 \
  --log-every 50 \
  --model-out dqn_flappy_m1_v2.pt
```

Run with checkpoint output (your command):

```bash
uv run python -m flappy_rl_tutorial.train \
  --episodes 3500 \
  --max-steps 3000 \
  --batch-size 128 \
  --lr 0.0007 \
  --eps-start 1.0 \
  --eps-end 0.02 \
  --eps-decay-steps 3000 \
  --target-update-interval 20 \
  --log-every 50 \
  --model-out dqn_flappy_m1_v2.pt \
  --checkpoint-out dqn_flappy_m1_v2_checkpoint.pt
```

Why this is better:

- in this trainer, epsilon decays per episode, so `--eps-decay-steps 3000` reaches low exploration much sooner
- lower learning rate is more stable than aggressive settings
- longer run gives time for reliable policy improvement

Stopping safely:

- press `Ctrl+C` to stop training
- current model is auto-saved to `--model-out`
- full training checkpoint is auto-saved to `--checkpoint-out` (default `dqn_flappy_checkpoint.pt`)

Continue training from a previous checkpoint:

```bash
uv run python -m flappy_rl_tutorial.train \
  --resume-from dqn_flappy_checkpoint.pt \
  --episodes 3000 \
  --model-out dqn_flappy_m1_v2.pt \
  --checkpoint-out dqn_flappy_checkpoint.pt
```

Note:

- `--episodes` here means additional episodes to run after loading the checkpoint
- if needed, add `--no-resume-optimizer` to resume only model weights

Run pygame in human mode:

```bash
uv run python -m flappy_rl_tutorial.play --mode human
```

Run pygame in AI mode using a trained checkpoint:

```bash
uv run python -m flappy_rl_tutorial.play --mode ai --model-path dqn_flappy_m1_v2.pt --auto-reset
```

## 3) RL concept behind the code

The DQN learns an approximation of:

`Q(s, a) = expected discounted return when taking action a in state s`

Training target:

`target = r + gamma * max_a' Q_target(s_next, a')` (if not done)

Loss:

`Huber(Q_online(s, a), target)`

Key ideas:

- online network learns every optimization step
- target network is updated slowly for stability
- replay buffer breaks time correlation by random sampling
- epsilon-greedy balances exploration and exploitation

## 4) File map

- `src/flappy_rl_tutorial/env.py`: small Flappy-like environment and reward rules
- `src/flappy_rl_tutorial/model.py`: DQN MLP
- `src/flappy_rl_tutorial/train.py`: replay buffer and training loop
- `src/flappy_rl_tutorial/play.py`: pygame game window for human or AI play

## 5) Suggested learning path

1. Read `env.py` and understand state and reward.
2. Read `choose_action()` in `train.py` for exploration.
3. Read Bellman target logic in the optimization block.
4. Train with defaults.
5. Experiment with one hyperparameter at a time:
   - `--gamma`
   - `--lr`
   - `--eps-decay-steps`
   - `--target-update-interval`
6. Open pygame in AI mode and watch policy behavior end-to-end.

## 6) Try these experiments

### A) Less exploration

```bash
uv run python -m flappy_rl_tutorial.train --episodes 300 --eps-end 0.2 --eps-decay-steps 2000
```

### B) More stable but slower updates

```bash
uv run python -m flappy_rl_tutorial.train --episodes 500 --target-update-interval 50
```

### C) Faster but riskier learning rate

```bash
uv run python -m flappy_rl_tutorial.train --episodes 300 --lr 0.003
```

## 7) Next upgrades

- switch to Double DQN
- add prioritized replay
- replace MLP state with image-based CNN input
