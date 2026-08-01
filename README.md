# Kidney Tumor Classification using Deep Learning

## Project Overview
This project focuses on classifying kidney CT scan images using a deep learning model. The goal is to build an end-to-end image classification pipeline with model training, evaluation, experiment tracking, and deployment support.

## Tech Stack
- Python
- TensorFlow / Keras
- MLflow
- DVC
- Flask
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Project Workflow
1. Data ingestion
2. Data preprocessing
3. Model training
4. Model evaluation
5. Experiment tracking with MLflow
6. Pipeline tracking with DVC
7. Web app deployment using Flask

## How to Run

## How to run?

### STEPS:

Clone the repository

```bash
git clone https://github.com/Abhitar3/Deep-Learning-Kidney-Tumor-Classification-.git
```

```bash
cd Deep-Learning-Kidney-Tumor-Classification-
```

### STEP 01 - Create a Python virtual environment after opening the repository

For this project, we are using a normal Python virtual environment instead of conda.

```powershell
python -m venv kidneyclassification
```

Activate the environment:

```powershell
.\kidneyclassification\Scripts\activate
```

After activation, your terminal should look like this:

```powershell
(kidneyclassification) PS C:\Users\abhin\Desktop\Mlops\Deep-Learning-Kidney-Tumor-Classification->
```

### STEP 02 - Upgrade pip, setuptools, and wheel

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### STEP 03 - Install the requirements

```powershell
pip install -r requirements.txt
```

### Important Windows note for TensorFlow

If TensorFlow fails with a long path error, open PowerShell as Administrator and run:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then restart VS Code or your terminal, activate the environment again, and rerun:

```powershell
pip install -r requirements.txt
```

### STEP 04 - Run Jupyter Notebook

```powershell
jupyter notebook
```

Open:

```text
research/trials.ipynb
```

### STEP 05 - Run MLflow UI

```powershell
mlflow ui
```

Open this URL in your browser:

```text
http://127.0.0.1:5000
```

### Project package

The project package name is:

```text
cnnclassifier
```

Because `requirements.txt` contains:

```txt
-e .
```

the local project will be installed in editable mode, so changes inside `src/cnnclassifier` can be imported directly while developing.
