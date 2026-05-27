##### Machine learning project for detecting fires early from sensor or image data.

### Setup

```bash
uv sync
.venv\Scripts\activate
```

### PyTorch (GPU)

1. Check CUDA version: `nvcc --version`
2. Match it in `pyproject.toml` — update the `[[tool.uv.index]]` URL (e.g. `cu128`, `cu130`) and run `uv lock`
3. `uv sync`, then verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Project structure

- `data/` - datasets
- `notebooks/` - experiments and training
- `models/` - saved models
- `docs/` - project notes and report

