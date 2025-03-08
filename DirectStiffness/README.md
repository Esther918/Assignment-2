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

Test that the code is working with pytest:
```bash
pytest --cov=DirectStiffness tests/
```
To run the test:
```bash
pytest -v
```
Run code review examples:
```bash
python example1_1.py
python example1_2.py
python example2_1.py
python example2_2.py
```
Run graded review examples:
```bash
python example_postprocess1.py
python example_postprocess2.py
python example_postprocess3.py
python example_sectry1.py
python example_sectry2.py
python example_sectry3.py
```
Note: example_sectry#.py are files of examples for second-try graded review.

The figure file name : structure_deformed_shape.png
