# KoopmanNet: An Efficient Model-Free Deep Koopman Kalman Filter for Trajectory Tracking

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)

This repository provides the official PyTorch implementation of the paper:  
> **"KoopmanNet: An Efficient Model-Free Deep Koopman Kalman Filter for Trajectory Tracking"**  
> *(Under Review)*

To promote research transparency and ensure full reproducibility, we release the core codebase and synthetic evaluation pipelines.

---

## 📌 Introduction

Accurate state estimation under severe nonlinearities, unmodeled disturbances, and unknown dynamic state-space models (SSMs) is of paramount importance in autonomous aerial tracking and aerospace engineering. 

**KoopmanNet** addresses these challenges by marrying **data-driven Koopman operator theory** with deep recurrent neural network architectures.


---

## 🔬 Benchmark Experiments

This repository includes the complete data generation, training, and benchmarking pipelines for the two representative synthetic benchmarks evaluated in the paper:
- **Synthetic Sinusoidal Model**: Evaluates tracking performance under nonlinear transitions, polynomial observations, and model mismatches.
- **Chaotic Lorenz Attractor**: Tests the model on highly nonlinear, chaotic trajectories under both matched and rotational observation model mismatches.

### 🚁 Note on the Real-World UZH-FPV Drone Dataset
For the real-world quadrotor flight tracking experiments (Section IV-D) on the **UZH-FPV Drone Racing Dataset**, we adopted the pre-processing setup and constant-acceleration model framework established in prior literature. To reproduce those experiments, readers and researchers are strongly encouraged to refer to:
- **Paper**: [*MAML-KalmanNet: A neural network-assisted Kalman filter based on model-agnostic meta-learning*](https://ieeexplore.ieee.org/document/10883047) (IEEE Transactions on Signal Processing)
- **Official Repository**: [ShanLi-2000/MAML-KalmanNet](https://github.com/ShanLi-2000/MAML-KalmanNet)

---

## 📂 Repository Structure & Module Overview

The core algorithmic package is organized under the `GSSFiltering/` directory:

```text
.
├── config.ini                   
├── GSSFiltering/               
│   ├── __init__.py
│   ├── dnn.py                   
│   ├── filtering.py             
│   ├── model.py                 
│   ├── tester.py                
│   └── trainer.py               
├── main_sinusoidal.py           
├── main_lorenz.py               
└── README.md
```
### Module Descriptions (`GSSFiltering/`)

| File Name | Functionality & Description |
| :--- | :--- |
| **`dnn.py`** | Defines the core neural network architecture (`OnlineKoopmanKNet`), including the lifting encoder, learnable linear Koopman operators, and the recurrent Kalman Gain network. |
| **`filtering.py`** | Implements the online filtering algorithm (`DataDriven_Koopman_Kalman_Filter`) that executes the recursive state prediction and measurement update loop. |
| **`model.py`** | Provides the state-space dynamic systems (Sinusoidal and Lorenz models, both matched and mismatched) along with synthetic trajectory data generators under various noise settings. |
| **`trainer.py`** | Handles the end-to-end model training process, including batch data loading, multi-task loss computation, parameter optimization, and periodic model checkpointing. |
| **`tester.py`** | Evaluates the trained model on validation and test sets, measuring tracking accuracy (MSE in dB) and per-step inference runtime. |
---

## 🚀 Quick Start

### 1. Requirements

Ensure you have Python 3.8+ and PyTorch installed:
```bash
pip install torch numpy scipy matplotlib
```

### 2. Configuration

Hyperparameters, including the Koopman lifting dimension $L$, sequence lengths, and noise settings, are specified in `config.ini`:
```ini
[Train]
train_iter = 300
valid_period = 25
batch_size = 1

[Train.Koopman]
learning_rate = 1e-3

[SyntheticSinusoidal]
n_features = 12
...
```

### 3. Running Experiments

- **Sinusoidal Experiment**:
  ```bash
  python main_sinusoidal.py
  ```
- **Lorenz Attractor Experiment (with 3D Trajectory Visualization)**:
  ```bash
  python main_lorenz.py
  ```

---

## 📜 Citation

If you find this codebase or our research useful in your work, please consider citing our work once published:

```bibtex
@article{koopmannet2026,
  title   = {KoopmanNet: An Efficient Model-Free Deep Koopman Kalman Filter for Trajectory Tracking},
  note    = {Under Review},
  year    = {2026}
}
```
