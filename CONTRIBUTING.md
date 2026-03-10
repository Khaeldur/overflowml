# Contributing to OverflowML

## Setup

```bash
git clone https://github.com/Khaeldur/overflowml.git
cd overflowml
pip install -e ".[dev,all]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests mock hardware detection and don't require a GPU.

## Code Style

- Line length: 100 (enforced by ruff)
- Follow existing patterns in the codebase
- No unnecessary comments or docstrings
- Keep it simple

```bash
ruff check overflowml/
```

## Adding a New Strategy

1. Add the strategy logic to `strategy.py` in `pick_strategy()`
2. Add corresponding tests in `tests/test_strategy.py`
3. If it requires new hardware detection, update `detect.py`
4. Update the benchmark models in `cli.py` if relevant

## Adding Hardware Support

1. Add a detection function in `detect.py` (e.g., `_detect_rocm()`)
2. Add the accelerator to the `Accelerator` enum
3. Update `detect_hardware()` priority order
4. Add strategy handling in `strategy.py`
5. Add tests

## Reporting Issues

Include:
- GPU model and VRAM
- System RAM
- OS (Windows/Linux/macOS)
- Python version
- `overflowml detect` output
- The model you're trying to load

## Pull Requests

- One logical change per PR
- Include tests for new functionality
- Run `pytest` and `ruff check` before submitting
- Describe what and why in the PR description
