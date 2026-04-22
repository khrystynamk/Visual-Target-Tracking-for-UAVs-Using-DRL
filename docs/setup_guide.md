# Setup Guide

Complete setup instructions for the Visual Target Tracking for UAVs Using DRL project.

## Prerequisites

- macOS with Xcode Command Line Tools (`xcode-select --install`)
- Conda (Miniconda or Anaconda)
- Unreal Engine 4.27 (built from source for AirSim)
- AirSim plugin (built with UE 4.27)
- Git

## 1. Clone the Repository

```bash
git clone https://github.com/khrystynamk/Visual-Target-Tracking-for-UAVs-Using-DRL.git
cd Visual-Target-Tracking-for-UAVs-Using-DRL
```

## 2. Create Conda Environment

```bash
conda create -n vtt python=3.10 -y
conda activate vtt
```

## 3. Install PyTorch

```bash
pip install torch torchvision
```

## 4. Install AirSim Python Client

AirSim requires special installation due to build dependencies:

```bash
pip install msgpack-rpc-python
pip install --no-build-isolation airsim
```

## 5. Install DeFM

DeFM provides the pretrained depth feature extractor:

```bash
pip install huggingface_hub omegaconf
git clone https://github.com/leggedrobotics/defm.git
cd defm
pip install -e . --no-deps
```

## 6. Install the Project

```bash
cd ~/Documents/Thesis/Visual-Target-Tracking-for-UAVs-Using-DRL
pip install -e .
```

This installs the `vtt` package in editable mode along with all remaining dependencies (gymnasium, stable-baselines3, wandb, opencv, etc.).

## 6. AirSim Settings

Copy the project's AirSim settings to the location AirSim reads from:

```bash
cp configs/airsim/settings.json ~/Documents/AirSim/settings.json
```

This configures:
- Two vehicles: `Drone1` (tracker) and `Target`
- Front-center camera at 224x224 with 90-degree FOV
- Depth (DepthPerspective, DepthVis) and RGB image types
- Target spawns 3m in front of the tracker

**Important:** AirSim reads settings from `~/Documents/AirSim/settings.json` on launch. Any changes to `configs/airsim/settings.json` must be copied there and UE4 must be restarted.

## 7. Launch the Simulator

### Using the Blocks environment (without opening the Editor)

```bash
/Users/kmysak/UE_4.27/Engine/Binaries/Mac/UE4Editor.app/Contents/MacOS/UE4Editor \
  /Users/kmysak/Documents/Thesis/AirSim/Unreal/Environments/Blocks/Blocks.uproject \
  -game -log
```

## 8. Running the Project

### P-Controller Baseline

```bash
# With scripted trajectory (sinusoidal, circular, figure_eight, helix)
python scripts/run_baseline.py --trajectory sinusoidal --duration 15

# With keyboard-controlled target
python scripts/run_baseline.py --trajectory keyboard
```

### Train SAC Agent (Depth)

```bash
# Start training
python scripts/train.py --config configs/drl/sac_depth.yaml

# Without W&B logging
python scripts/train.py --config configs/drl/sac_depth.yaml --no-wandb

# Resume from checkpoint
python scripts/train.py --config configs/drl/sac_depth.yaml \
    --resume experiments/sac_depth/checkpoints/sac_5000_steps.zip
```

### Evaluate and Compare

```bash
python scripts/evaluate.py --model experiments/sac_depth/best/best_model.zip --duration 15
```

This runs both the P-controller baseline and the SAC agent on sinusoidal, circular, and figure-8 trajectories, generating comparison plots and metrics in `experiments/evaluation/`.

## 9. Weights & Biases (Optional)

For experiment tracking:

```bash
wandb login
```

Training curves, episode rewards, and metrics are logged automatically when W&B is enabled.

## 10. Project Structure

```
Visual-Target-Tracking-for-UAVs-Using-DRL/
├── configs/
│   ├── airsim/settings.json       # AirSim vehicle/camera config
│   └── drl/sac_depth.yaml         # SAC training hyperparameters
├── scripts/
│   ├── train.py                   # Train SAC agent
│   ├── run_baseline.py            # Run P-controller baseline
│   └── evaluate.py                # Compare baseline vs SAC
├── src/vtt/
│   ├── constants.py               # Shared constants
│   ├── envs/tracking_env.py       # Gymnasium environment (AirSim)
│   ├── models/
│   │   ├── feature_extractors.py  # DepthResNet (ResNet-18 for depth)
│   │   └── asymmetric_policy.py   # Asymmetric SAC (actor/critic)
│   ├── control/p_controller.py    # P-controller baseline
│   ├── target/
│   │   ├── trajectory_follower.py # Target follows scripted trajectories
│   │   └── keyboard_controller.py # Manual keyboard control
│   └── metrics/
│       ├── tracker_metrics.py     # Detection rate, RMSE, loss events
│       ├── scripted_trajectories.py # Sinusoidal, circular, figure-8
│       └── trajectory_comparison.py # 3D plots, error computation
└── experiments/                   # Training outputs (gitignored)
```
