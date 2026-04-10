# TD3 Robosuite — Panda Door Opening

Train a simulated Panda robot arm to open a door using **TD3 (Twin Delayed DDPG)** and **robosuite**.

Based on the [6-part YouTube series](https://www.youtube.com/watch?v=z1Lnlw2m8dg) by MQuantran, updated for robosuite 1.4+ and gymnasium.

---

## Setup

### 1. Prerequisites

- Python 3.11 (via [Anaconda](https://www.anaconda.com/))
- NVIDIA GPU with CUDA 12.8 (CPU works too, just slower)
- Windows 10/11 or Linux

### 2. Create environment and install

```bash
conda create -n td3_door python=3.11 -y
conda activate td3_door
pip install -r requirements.txt
```

> **Windows only:** Create `C:\tmp\` if it doesn't exist.  
> If you get an EGL graphics error, open the installed robosuite package and in  
> `robosuite/utils/binding_utils.py` line ~43 change `"egl"` → `"wgl"`.

### 3. Verify everything works

```bash
python check_setup.py
```

All checks should show `✓ PASS` before you start training.

---

## Usage

### Train

```bash
python main.py
```

Training runs for 10 000 episodes. Models are saved to `tmp/td3/` every 10 episodes and automatically reloaded on restart, so you can stop and resume at any time.

### Monitor training

```bash
tensorboard --logdir logs --port 6006
```

Open `http://localhost:6006` in your browser. Look for the score trend rising above 0 after ~1 000 episodes.

### Watch the robot (after some training)

```bash
python test.py
```

Runs 3 episodes with the renderer open. Requires a display.

---

## Project structure

```
td3_door/
├── buffer.py       # Experience replay buffer
├── networks.py     # Actor and Critic neural networks (PyTorch)
├── td3_torch.py    # TD3 Agent (6 networks, Polyak averaging)
├── main.py         # Training loop
├── test.py         # Visual test script
├── check_setup.py  # Pre-flight checks
└── requirements.txt
```

## Algorithm overview

TD3 (Twin Delayed DDPG) key ideas:
- **Twin critics** — take the minimum Q-value to reduce overestimation bias
- **Delayed actor updates** — update the actor less frequently than critics
- **Target policy smoothing** — add clipped noise to target actions for regularisation
- **Polyak averaging** — soft-update target networks each step

## Hyperparameters

| Parameter | Value |
|---|---|
| Actor LR | 0.001 |
| Critic LR | 0.001 |
| Batch size | 128 |
| Replay buffer | 1 000 000 |
| Discount γ | 0.99 |
| Polyak τ | 0.005 |
| Episode horizon | 300 steps |
| Warm-up steps | 1 000 |
