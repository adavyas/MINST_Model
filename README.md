# Fashion-MNIST Classification (PyTorch + Apple MPS)

Train a compact CNN on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset with **≥90% accuracy** using PyTorch.  
Optimized to run efficiently on **Apple Silicon (MPS)** with CPU fallback.

---

## 📦 Features
- **Apple MPS support**: Accelerated training on Apple Silicon GPUs (falls back to CPU automatically).
- **Compact CNN (~0.8M params)**: Achieves ~92% test accuracy in ~40 epochs.
- **Regularization tricks**: Label smoothing, dropout, weight decay, gradient clipping.
- **Modern optimizations**:
  - **AdamW** optimizer with proper weight-decay hygiene.
  - **OneCycleLR** learning rate schedule.
  - **EMA (Exponential Moving Average)** of weights for stable evaluation.
- **Data augmentations**: Random crop, flip, rotation, and erasing to improve generalization.
- **Checkpointing**: Best validation model saved automatically.

---

## 🚀 Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/adavyas/fashion-mnist-mps.git
cd fashion-mnist-mps
python3 -m venv env
source env/bin/activate   # or `source env/bin/activate.fish` if using fish shell
pip install -r requirements.txt
