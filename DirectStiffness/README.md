# Direct Stiffness Model
Set up a conda environment:
```bash
conda create --name DirectStiffness-env python=3.12
```
Activate the environment:
```bash
conda activate DirectStiffness-env
```
Double check that python is version 3.12:
```bash
python --version
```
Ensure that pip is using the most up to date:
```bash
pip install --upgrade pip setuptools wheel
```
Ensure pyproject.toml exists and install the package:
If pyproject.toml is present, run:
```bash
pip install -e .
```
Note: make sure you are in the project root directory and that pyproject.toml exists.

Test that the code is working with pytest (pytest is not ready yet and will be fixed later, please skip to Run example):
```bash
pytest --cov=DirectStiffness tests/
```
To run the test:
```bash
pytest -v
```
Run example (this example is from example2):
```bash
python example_postprocess.py
```
