# QSVM QOSF Task 4 – Quantum Support Vector Machine on Iris Dataset

##  Project Overview
This project implements and evaluates **Quantum Support Vector Machines (QSVMs)** for a binary classification problem using the Iris dataset. The goal is to explore how different quantum circuit architectures affect classification performance, decision boundaries, and expressibility.  

This work is part of the QOSF Mentorship **Task 4: QSVM** screening task.

---

##  Folder Structure
```
QSVM_QOSF_Task4/
│
├── README.md # Project overview, methods, and results
├── qsvm_task4.ipynb # Main Jupyter notebook with code, figures, and explanations
├── src/ # Source code modules
│ ├── data_preprocessing.py # Data loading, filtering, normalization, splitting
│ ├── qsvm_models.py # Quantum circuit definitions, QSVM kernel computation, training
│ └── utils.py # Helper functions for plotting and evaluation
├── figures/ # Generated figures and visualizations
│ ├── iris_distribution.png
│ ├── qsvm_train_boundary.png
│ ├── qsvm_test_boundary.png
│ ├── qsvm_alt_boundary.png
│ └── performance_comparison.png
└── requirements.txt # Python dependencies for reproducibility
```


---

##  Methodology

### 1️) Data Preparation
- Selected two classes from the Iris dataset: **Setosa** and **Versicolor** for binary classification.  
- Normalized features using `StandardScaler`.  
- Split dataset into **training (80%)** and **testing (20%)** sets.  
- Visualized feature distribution to explore separability.

### 2️) Classical Baseline
- Trained a **classical SVM** on the normalized data to serve as a performance benchmark.  
- Evaluated train/test accuracy and plotted decision boundaries.  

### 3️) Quantum Models (QSVM)
Two distinct quantum circuits were implemented:

#### **QSVM Model 1 – Simple RY + CX Circuit**
- 2-qubit circuit, parameterized with `RY` rotations.  
- Single entangling layer using `CX`.  
- Shallow and robust; suitable for linearly separable data.

#### **QSVM Model 2 – RZ + CX + Layered Circuit**
- 2-qubit circuit with `RZ` rotations and additional entanglement layers.  
- Deeper and more expressive, able to capture more complex boundaries.  
- Slightly higher risk of overfitting small datasets.

- Computed **Fidelity Quantum Kernel** for both circuits.  
- Trained **SVM with precomputed kernel**.  
- Evaluated train/test accuracy, confusion matrices, and plotted decision boundaries.

---

##  Results & Analysis

- ### Training and Testing Accuracy
| Model            | Train Accuracy (%) | Test Accuracy (%) |
|-----------------|-----------------|----------------|
| QSVM Model 1     | 91.25           | 85.0           |
| QSVM Model 2     | 81.25           | 85.0           |
| Classical SVM    | 98.75           | 100.0          |

### Observations
- The **classical SVM outperforms both QSVM models** on this dataset.  
- Reason: The dataset is small and linearly separable; classical linear SVM is sufficient.  
- QSVM Model 1 is shallow and robust, Model 2 is more expressive but slightly underfits on training data.  
- Quantum kernels provide **nonlinear feature mapping**, which is more useful for complex or overlapping datasets.  
- This demonstrates that **quantum models may not always outperform classical models on simple problems**, but they provide a framework for scaling to more complex scenarios.

### Figures
- `iris_distribution.png`: Scatter plot of features.  
- `qsvm_train_boundary.png` / `qsvm_test_boundary.png`: Decision boundaries for QSVM Model 1.  
- `qsvm_alt_boundary.png`: Decision boundary for QSVM Model 2.  
- `performance_comparison.png`: Bar chart comparing models’ accuracy.

---

##  Requirements
Python 3.9 with the following packages:
```
qiskit
qiskit-machine-learning
scikit-learn
numpy
matplotlib
```

Install via:
```bash
pip install -r requirements.txt
```

##  References

- [Qiskit Machine Learning Documentation](https://qiskit.org/documentation/machine-learning/)
- [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- [QOSF Mentorship Task 4 Description](https://qosf.slack.com/archives/C019UEZRCM9)
