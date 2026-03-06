# A Monocular Dense SLAM Framework for 3D Gaussian Rendering in Long-Term Complex Environments

<p align="center">
  <b>Yinchu Wang, Ximeng Cheng, Xiaobo Lu*</b><br>
  School of Automation, Southeast University, Nanjing, China<br>
  Key Laboratory of Measurement and Control of Complex Systems of Engineering, Ministry of Education
</p>

---

> **Abstract:** We propose a novel monocular SLAM algorithm tailored for long-term, challenging scenarios. Our framework integrates an attention-based feature correlation module with Deformable Convolutional GRU (DeformConv-GRU) for robust inter-frame correspondence, a Depth Estimation Bundle Adjustment (DEBA) module that fuses DepthPro-based monocular depth predictions with motion-based depth, and an Adaptive Global Bundle Adjustment (AGBA) strategy that efficiently optimizes accumulated poses via sparse graph construction. Our method achieves state-of-the-art average ATE on TUM (36.84% reduction) and EuRoC (9.1% reduction) benchmarks compared to leading learning-based methods. We further validate downstream 3DGS rendering quality (PSNR 27.93) on custom complex-environment datasets.

---

## Architecture Overview

```
Input RGB Sequence
        │
        ▼
┌─────────────────────────────────┐
│  Feature Correlation Network     │
│  ┌─────────────┐  ┌──────────┐  │
│  │ BasicEncoder │──│ Self-Attn│  │   Shared convolutional encoder
│  │ (fnet/cnet)  │  └──────────┘  │   with residual blocks + self-attention
│  └──────┬───────┘                │
│         │                        │
│  ┌──────▼───────┐                │
│  │ Cross-Attn   │                │   Inter-frame feature correspondence
│  └──────┬───────┘                │
│         │                        │
│  ┌──────▼───────┐                │
│  │ 4D Correlation│               │   Multi-scale correlation pyramid
│  │ Volume        │               │
│  └──────┬───────┘                │
│         │                        │
│  ┌──────▼────────────┐           │
│  │ DeformConv-GRU    │           │   Adaptive receptive field GRU
│  │ → Displacement M  │           │   → pixel-wise displacement + confidence
│  │ → Confidence   W  │           │
│  └───────────────────┘           │
└──────────────┬──────────────────┘
               │
        ┌──────▼──────┐
        │    DEBA      │    Depth Estimation Bundle Adjustment
        │  DepthPro ──►│    Log-space scale alignment +
        │  d_e + d_c   │    confidence-weighted depth fusion
        └──────┬───────┘
               │
        ┌──────▼──────┐
        │    AGBA      │    Adaptive Global Bundle Adjustment
        │  Chebyshev   │    Residual-triggered + sparse graph
        │  + 2-hop     │    (13× faster, 55% less GPU memory)
        └──────┬───────┘
               │
        ┌──────▼──────┐
        │  Poses +     │
        │  Dense PCD   │──► TSDF Mesh / 3DGS Rendering
        └─────────────┘
```

## Key Contributions

| Module | Description | Paper Section |
|--------|-------------|---------------|
| **DeformConv-GRU** | Deformable convolutions replace fixed-grid GRU convolutions, enabling adaptive receptive fields for complex motion | §3.1 |
| **Cross-Attention** | Bidirectional cross-attention between frame pairs for non-local inter-frame correspondence | §3.1 |
| **DEBA** | Fuses DepthPro (VFM) absolute depth with motion-based depth via log-space cubic spline alignment | §3.2 |
| **AGBA** | Chebyshev-distance graph sparsification with two-hop suppression, residual-based adaptive triggering | §3.3 |

---

## Installation

### Prerequisites

- Ubuntu 18.04+ / Windows 10+ / macOS
- Python 3.9+
- CUDA 11.7+ (GPU with ≥8GB VRAM recommended)
- Conda (recommended)

### Step 1: Clone the Repository

```bash
git clone --recursive https://github.com/<your-repo>/monocular-dense-slam.git
cd monocular-dense-slam
```

If you already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

### Step 2: Create Conda Environment

```bash
conda create -n slam python=3.9
conda activate slam
```

### Step 3: Install PyTorch

Install PyTorch matching your CUDA version (see [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)):

```bash
# Example: CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Example: CUDA 12.1
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install DROID-SLAM Backend (C++/CUDA extensions)

```bash
cd modules/droid_slam
python setup.py install
cd ../..
```

### Step 6: Integrate DepthPro

DepthPro is used as our Visual Foundation Model (VFM) for monocular metric depth estimation in the DEBA module.

```bash
# Clone DepthPro into modules/
cd modules
git clone https://github.com/apple/ml-depth-pro.git depth_pro
cd depth_pro
pip install -e .
cd ../..
```

The DepthPro integration works as follows:
- `modules/metric.py` → `DepthProEstimator` class wraps the DepthPro model
- `depth.py` → runs DepthPro inference on an image directory
- The DEBA module (`modules/droid_core/deba.py`) consumes DepthPro outputs as `d_e` (VFM depth estimate) and fuses them with motion-based depth `d_c`

**DepthPro Dual-ViT Architecture:**
- Global ViT encoder captures scene-level context
- Multi-resolution patch ViT extracts fine-grained local details
- HFOV module predicts focal length for metric-scale projection

### Step 7: Verify Installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from modules import Droid, DepthPro, RGBDFusion; print('All modules loaded')"
```

---

## Pretrained Weights

```bash
python download_models.py
```

This downloads:
| Weight | File | Description |
|--------|------|-------------|
| DROID-SLAM | `weights/droid.pth` | Base DROID-SLAM network weights |
| DepthPro | `weights/depth_pro.pt` | DepthPro metric depth estimation |

> **Note:** Our fine-tuned weights (with DeformConv-GRU + attention modules) will be released upon paper acceptance. The current weights are from the base models. See [TODO](#todo) for release timeline.

---

## Quick Start

### Full Pipeline (Video → 3D Reconstruction)

```bash
python inference.py \
    --input path/to/video.mp4 \
    --output results/my_scene \
    --viz
```

### Full Pipeline (Image Directory → 3D Reconstruction)

```bash
python inference.py \
    --input path/to/images/ \
    --output results/my_scene \
    --intr path/to/intrinsic.txt \
    --viz
```

### Export to COLMAP Format (for 3DGS/PGSR)

```bash
python inference.py \
    --input path/to/images/ \
    --output results/my_scene \
    --intr path/to/intrinsic.txt \
    --export-colmap
```

---

## Training

Our network is trained end-to-end with the combined loss:

**L = w₁ · L_pose + w₂ · L_res**

where:
- **L_pose** = Σᵢ ‖log_SE3(Tᵢ⁻¹ · Gᵢ)‖₂ (pose-space loss on SE(3) manifold)
- **L_res** = Σₖ γⁿ⁻ᵏ⁻¹ ‖Rₖ‖ (temporal residual loss with exponential decay)

### Training on TUM-RGBD

```bash
python train.py \
    --dataset_dir ./data/TUM \
    --dataset_type tum \
    --epochs 25 \
    --batch_size 1 \
    --lr 2.5e-4 \
    --w_pose 1.0 \
    --w_res 0.1 \
    --gamma 0.9 \
    --num_steps 12 \
    --output_dir ./checkpoints
```

### Training on EuRoC

```bash
python train.py \
    --dataset_dir ./data/EuRoC \
    --dataset_type euroc \
    --epochs 25 \
    --batch_size 1 \
    --lr 2.5e-4 \
    --ckpt ./weights/droid.pth \
    --output_dir ./checkpoints
```

### Training on TartanAir

```bash
python train.py \
    --dataset_dir ./data/TartanAir \
    --dataset_type tartanair \
    --epochs 25 \
    --clip_len 7 \
    --stride 2 \
    --output_dir ./checkpoints
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 2.5e-4 | Learning rate (AdamW) |
| `--epochs` | 25 | Number of training epochs |
| `--batch_size` | 1 | Batch size (limited by GPU memory) |
| `--clip_len` | 7 | Frames per training clip |
| `--num_steps` | 12 | GRU update iterations |
| `--w_pose` | 1.0 | Pose loss weight (w₁) |
| `--w_res` | 0.1 | Residual loss weight (w₂) |
| `--gamma` | 0.9 | Temporal decay factor |

Monitor training with TensorBoard:
```bash
tensorboard --logdir ./checkpoints
```

---

## Inference

### Step-by-step Execution

For finer control, run each pipeline stage independently:

#### Step 1: Depth Estimation (DepthPro)

```bash
python depth.py \
    --images path/to/rgb/ \
    --out path/to/output/depth \
    --intr path/to/intrinsic.txt \
    --dmax 500.0 \
    --out-colormap  # optional: save depth visualization
```

#### Step 2: Camera Pose Estimation (DROID-SLAM + DEBA + AGBA)

```bash
python slam.py \
    --images path/to/rgb/ \
    --depth path/to/output/depth \
    --intr path/to/intrinsic.txt \
    --out-poses path/to/output/poses \
    --out-traj path/to/output/trajectory.txt \
    --global-ba-frontend 90 \
    --viz
```

#### Step 3: Mesh Reconstruction (TSDF Fusion)

```bash
python mesh.py \
    --images path/to/rgb/ \
    --depth path/to/output/depth \
    --poses path/to/output/poses \
    --intr path/to/intrinsic.txt \
    --save path/to/output/mesh.ply \
    --voxel-length 0.02
```

#### Step 4 (Optional): Export to COLMAP for 3DGS

```bash
python inference.py \
    --input path/to/rgb/ \
    --output path/to/output \
    --intr path/to/intrinsic.txt \
    --step colmap
```

### Intrinsic File Format

The intrinsic file contains 4 values (one per line):
```
fx
fy
cx
cy
```

If no intrinsic is provided, it is estimated as:
- `fx = fy = max(image_width, image_height) × 1.2` (following COLMAP)
- `cx = image_width / 2`
- `cy = image_height / 2`

### Camera Calibration

```bash
python scripts/calib.py path/to/images_or_video \
    --pattern chessboard \
    --pattern-size 9 6 \
    --square-size 15 \
    --write intrinsic.txt
```

---

## Evaluation

### Single Trajectory Evaluation

```bash
python evaluate.py \
    --est results/trajectory.txt \
    --gt data/TUM/fr1_desk/groundtruth.txt \
    --dataset tum
```

### Batch Evaluation on TUM-RGBD

```bash
python evaluate.py \
    --dataset tum \
    --data_root ./data/TUM \
    --results_dir ./results/tum \
    --num_runs 5 \
    --save_json results/tum_results.json
```

### Batch Evaluation on EuRoC

```bash
python evaluate.py \
    --dataset euroc \
    --data_root ./data/EuRoC \
    --results_dir ./results/euroc \
    --num_runs 5 \
    --save_json results/euroc_results.json
```

### Run SLAM + Evaluate

```bash
python evaluate.py \
    --dataset tum \
    --data_root ./data/TUM \
    --results_dir ./results/tum \
    --run \
    --slam_ckpt ./weights/droid.pth \
    --depth_ckpt ./weights/depth_pro.pt \
    --num_runs 5
```

### Reported Results

**TUM-RGBD ATE RMSE (m):**

| Sequence | DROID-SLAM | Ours |
|----------|-----------|------|
| fr1/desk | 0.019 | **0.012** |
| fr1/desk2 | 0.040 | **0.025** |
| fr1/room | 0.046 | **0.024** |
| fr2/xyz | 0.002 | **0.002** |
| fr3/office | 0.035 | **0.022** |
| **Average** | **0.038** | **0.024** (↓36.84%) |

**EuRoC ATE RMSE (m):**

| Sequence | DROID-SLAM | Ours |
|----------|-----------|------|
| MH_01 | 0.013 | **0.010** |
| MH_02 | 0.014 | **0.010** |
| V1_01 | 0.012 | **0.010** |
| V2_01 | 0.017 | **0.015** |
| **Average** | **0.022** | **0.020** (↓9.1%) |

**AGBA Efficiency (Forest Dataset):**

| Method | ATE (m) | Avg. BA Time (s) | Peak GPU (GB) | Graph Edges |
|--------|---------|-------------------|---------------|-------------|
| Full BA | 3.95 | 4.82 | 15.2 | ~1.2M |
| **AGBA** | **3.32** | **0.35** (13× faster) | **6.8** (55% less) | **~95K** |

---

## 3DGS Rendering (Downstream Task)

Our SLAM outputs can be directly used for 3D Gaussian Splatting rendering:

```bash
# 1. Run our SLAM pipeline with COLMAP export
python inference.py \
    --input path/to/images/ \
    --output results/scene \
    --intr path/to/intrinsic.txt \
    --export-colmap

# 2. Feed into PGSR or other 3DGS frameworks
# The COLMAP-compatible output is in results/scene/colmap/sparse/0/
#   cameras.txt   - pinhole camera intrinsics
#   images.txt    - camera poses (world-to-camera)
#   points3D.txt  - empty (3DGS initializes from depth maps)
```

Our method achieved **PSNR 27.93** on the Milarepa Buddhist Pavilion scene with PGSR rendering.

---

## Project Structure

```
.
├── train.py              # Training script
├── inference.py          # Full inference pipeline
├── evaluate.py           # Evaluation on TUM/EuRoC/custom datasets
├── depth.py              # DepthPro depth estimation
├── slam.py               # DROID-SLAM with DEBA + AGBA
├── mesh.py               # TSDF mesh reconstruction
├── reconstruct.py        # Legacy one-command reconstruction
├── download_models.py    # Download pretrained weights
├── download_dataset.py   # Download example dataset
├── requirements.txt      # Python dependencies
├── weights/              # Model checkpoints
│   ├── droid.pth
│   └── depth_pro.pt
├── modules/
│   ├── __init__.py
│   ├── metric.py         # DepthPro estimator wrapper (DepthProEstimator)
│   ├── droid.py          # DROID-SLAM runner (Options, RGBDStream)
│   ├── fusion.py         # TSDF volume fusion + mesh extraction
│   ├── data.py           # PosedImageStream dataset
│   ├── utils.py          # Utilities (calibration, trajectory conversion)
│   ├── depth_pro/        # DepthPro submodule (Apple ml-depth-pro)
│   ├── droid_slam/       # DROID-SLAM C++/CUDA backend
│   └── droid_core/
│       ├── droid_net.py      # DroidNet: Feature Correlation + DeformConv-GRU
│       ├── droid.py          # Core Droid class
│       ├── droid_frontend.py # Frontend with DEBA integration
│       ├── droid_backend.py  # Backend with AGBA strategy
│       ├── deba.py           # DEBA module implementation
│       ├── depth_video.py    # Shared depth-pose video buffer
│       ├── factor_graph.py   # Factor graph for BA optimization
│       ├── motion_filter.py  # Motion-based keyframe filter
│       └── modules/
│           ├── extractor.py  # BasicEncoder + SelfAttention + CrossAttention
│           ├── gru.py        # DeformConv-GRU (DeformConvLayer + ConvGRU)
│           ├── corr.py       # 4D Correlation volume
│           └── clipping.py   # Gradient clipping utility
├── scripts/
│   ├── calib.py          # Camera calibration
│   ├── sample.py         # Video frame sampling
│   ├── undistort.py      # Image undistortion
│   └── viz_scene.py      # Scene visualization
└── rgbd_benchmark/       # TUM RGBD evaluation tools
    ├── evaluate_ate.py
    ├── evaluate_rpe.py
    └── associate.py
```

---

## Datasets

### TUM-RGBD

Download from [TUM RGB-D Benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset):
```bash
# Example: fr1/desk
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xzf rgbd_dataset_freiburg1_desk.tgz -C data/TUM/
```

### EuRoC MAV

Download from [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets):
```bash
# Example: MH_01_easy
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
unzip MH_01_easy.zip -d data/EuRoC/MH_01_easy/
```

### Forest Dataset

The Forest dataset by the MRS group at CTU Prague, a 19-minute UAV trajectory (~1.25 km) through sparse forest terrain. Download from [the dataset page](https://github.com/ctu-mrs/slam-datasets).

### Dronescapes

The Dronescapes dataset for aerial SLAM in complex environments. See [Dronescapes](https://github.com/ivandariomarcuzzo/dronescapes).

---

## TODO

- [x] Release core SLAM pipeline code
- [x] Release training / inference / evaluation scripts
- [x] DepthPro integration (DEBA module)
- [x] AGBA implementation
- [x] DeformConv-GRU implementation
- [x] Cross-attention feature correlation module
- [x] COLMAP export for 3DGS downstream tasks
- [ ] Release fine-tuned model weights (DeformConv-GRU + attention)
- [ ] Release DEBA-specific fine-tuned weights
- [ ] Pre-built Docker image
- [ ] ROS/ROS2 integration node
- [ ] Real-time demo with webcam
- [ ] Benchmark results reproduction scripts
- [ ] Detailed training guide with hyperparameter sweeps

> **Weights Release Timeline:** Model weights will be released upon paper acceptance. In the meantime, the base DROID-SLAM weights (`droid.pth`) and DepthPro weights (`depth_pro.pt`) can be used via `python download_models.py`.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2025monocular,
  title={A Monocular Dense SLAM Framework for 3D Gaussian Rendering in Long-Term Complex Environments},
  author={Wang, Yinchu and Cheng, Ximeng and Lu, Xiaobo},
  journal={},
  year={2025}
}
```

## Acknowledgements

This work was supported by the Frontier Technology R&D Program of Jiangsu (BF2024060), the National Natural Science Foundation of China under grant 62271143, and the Big Data Computing Center of Southeast University.

Our implementation builds upon the following excellent works:
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) — Base SLAM architecture
- [DepthPro](https://github.com/apple/ml-depth-pro) — Monocular metric depth estimation (VFM in DEBA)
- [RAFT](https://github.com/princeton-vl/RAFT) — Optical flow and correlation volume design
- [lietorch](https://github.com/princeton-vl/lietorch) — Lie group operations for SE3

## License

This project is released for academic research purposes. Please see individual submodule licenses for third-party components.
