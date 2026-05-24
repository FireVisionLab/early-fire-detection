##### Machine learning project for detecting fires early from sensor or image data.

### Setup

```bash
uv sync
.venv\Scripts\activate
```

### PyTorch (GPU)

Check your CUDA version first:

```bash
nvcc --version
```

Then install PyTorch for that version from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# For example CUDA 13.0
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Project structure

- `data/` - datasets
- `notebooks/` - experiments and training
- `models/` - saved models
- `docs/` - project notes and report

