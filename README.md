# EN4554 Assignment 2 — CNNs (CIFAR-10, PyTorch)

**Eashan S.G.S — 220148G**

Notebook: [`220148G_a02.ipynb`](220148G_a02.ipynb)

| Task | Topic |
|------|-------|
| 1 | Backpropagation from scratch (Linear, ReLU, softmax cross-entropy), gradient checks, vanishing gradients |
| 2 | SGD-momentum and Adam as `torch.optim.Optimizer` subclasses, quadratic-bowl trajectories, LR sensitivity |
| 3 | Regularization on 5k images (weight decay, dropout, flip+crop, cutout, mixup), plain vs residual nets (depth 8/20/56) |
| 4 | Best CNN with ≤ 500k parameters and ≤ 20 epochs, ablation, single test-set evaluation |

## Running
Open the notebook in Google Colab (Runtime → T4 GPU) and run all cells.
Results are cached to `results/` (or Google Drive on Colab), so an interrupted run can resume.

CIFAR-10 split: 45k train / 5k validation (seeded), per-channel normalization from training statistics.
The test set is used once, in Task 4.
