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

## 7. AirSim Settings

Copy the project's AirSim settings to the location AirSim reads from:

```bash
cp configs/airsim/settings.json ~/Documents/AirSim/settings.json
```

This configures:
- Two vehicles: `Drone1` (tracker) and `Target`
- Front-center camera at 224x224 with 90-degree FOV
- Depth (DepthPerspective, DepthVis) and RGB image types
- Target spawns 2m in front of the tracker

**Important:** AirSim reads settings from `~/Documents/AirSim/settings.json` on launch. Any changes to `configs/airsim/settings.json` must be copied there and UE4 must be restarted.

## 8. Launch the Simulator

### Using the Blocks environment (without opening the Editor)

```bash
/Users/kmysak/UE_4.27/Engine/Binaries/Mac/UE4Editor.app/Contents/MacOS/UE4Editor \
  /Users/kmysak/Documents/Thesis/AirSim/Unreal/Environments/Blocks/Blocks.uproject \
  -game -log
```

## 9. Running the Project

### P-Controller Baseline

```bash
# With scripted trajectory (sinusoidal, circular, figure_eight, helix)
python scripts/run_baseline.py --trajectory sinusoidal --duration 15

# With keyboard-controlled target
python scripts/run_baseline.py --trajectory keyboard
```

### Run Trained SAC Agent

```bash
python scripts/run_trained.py --model model.zip --trajectory sinusoidal --duration 15
```

### Train SAC Agent (Depth)

```bash
# Start training (single environment)
python scripts/train.py --config configs/drl/sac_depth.yaml

# With parallel environments
python scripts/train.py --config configs/drl/sac_depth.yaml --n-envs 6

# Without W&B logging
python scripts/train.py --config configs/drl/sac_depth.yaml --no-wandb

# Resume from checkpoint
python scripts/train.py --config configs/drl/sac_depth.yaml \
    --resume experiments/sac_depth/checkpoints/sac_5000_steps.zip

# Resume latest from R2 cloud storage
python scripts/train.py --config configs/drl/sac_depth.yaml --resume auto --r2-sync
```

### Evaluate and Compare (Baseline vs SAC)

```bash
# 5 runs per method (default)
python scripts/evaluate.py --model experiments/sac_depth/best/best_model.zip \
    --trajectory sinusoidal

# Custom number of runs and duration
python scripts/evaluate.py --model experiments/sac_depth/best/best_model.zip \
    --trajectory circular --runs 3 --duration 20

# With weather/lighting condition (depth robustness test)
python scripts/evaluate.py --model experiments/sac_depth/best/best_model.zip \
    --trajectory sinusoidal --condition rain
```

Available trajectories: `sinusoidal`, `circular`, `figure_eight`, `helix`

Available conditions: `default`, `rain`, `fog`, `dust`, `dawn`, `dawn_rain`

## 10. Weights & Biases (Optional)

For experiment tracking:

```bash
wandb login
```

Training curves and episode rewards are logged automatically when W&B is enabled.

## 11. Project Structure

```
Visual-Target-Tracking-for-UAVs-Using-DRL/
├── configs/
│   ├── airsim/settings.json          # AirSim vehicle/camera config
│   └── drl/sac_depth.yaml            # SAC training hyperparameters
├── docs/
│   ├── setup_guide.md
│   └── airsim_setup_guide.md
├── scripts/
│   ├── train.py                      # Train SAC agent
│   ├── run_baseline.py               # Run P-controller baseline
│   ├── run_trained.py                # Run trained SAC agent
│   ├── evaluate.py                   # Compare baseline vs SAC
│   ├── bootstrap_vastai.sh           # Vast.ai cloud bootstrap
│   ├── launch_airsim_fleet.sh        # Launch parallel AirSim instances
├── src/vtt/
│   ├── constants.py                  # Shared constants
│   ├── envs/
│   │   └── tracking_env.py           # Gymnasium environment (AirSim)
│   ├── models/
│   │   ├── feature_extractors.py     # DepthResNet (ResNet-18 + DeFM)
│   │   └── asymmetric_policy.py      # Asymmetric SAC (actor/critic)
│   ├── control/
│   │   └── p_controller.py           # P-controller baseline
│   ├── target/
│   │   ├── trajectory_follower.py    # Target follows scripted trajectories
│   │   └── keyboard_controller.py    # Manual keyboard control
│   ├── metrics/
│   │   ├── constants.py              # Trajectory presets (train + eval)
│   │   ├── tracker_metrics.py        # Detection rate, RMSE, loss events
│   │   ├── scripted_trajectories.py  # Sinusoidal, circular, figure-8, helix
│   │   └── trajectory_comparison.py  # 3D plots, error computation
│   ├── callbacks/
│   │   ├── image_monitor.py          # Depth image logging callback
│   │   └── r2_sync.py                # R2 cloud sync callback
│   └── utils/
│       ├── camera_helpers.py         # Camera capture, detection, bbox
│       └── common_utils.py           # Shared utility functions
```
