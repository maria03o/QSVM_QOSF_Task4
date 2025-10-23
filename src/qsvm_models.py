

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


def qsvm_model_1():
    """Simple 2-qubit RY + CX circuit."""
    x1, x2 = Parameter("x1"), Parameter("x2")
    qc = QuantumCircuit(2)
    qc.ry(x1, 0)
    qc.ry(x2, 1)
    qc.cx(0, 1)
    qc.ry(x1, 0)
    qc.ry(x2, 1)
    return qc


def qsvm_model_2():
    """More expressive 2-qubit RZ + RY + CX layered circuit."""
    x1, x2 = Parameter("x1"), Parameter("x2")
    qc = QuantumCircuit(2)
    qc.ry(x1, 0)
    qc.rz(x2, 0)
    qc.ry(x2, 1)
    qc.cx(0, 1)
    qc.rz(x1, 1)
    qc.cx(1, 0)
    return qc



def train_qsvm(feature_map, x_train, y_train):
    """
    Compute fidelity kernel and train SVM with precomputed kernel.
    """
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    kernel_train = quantum_kernel.evaluate(x_vec=x_train, y_vec=x_train)

    model = SVC(kernel="precomputed")
    model.fit(kernel_train, y_train)
    return model, quantum_kernel


def evaluate_qsvm(model, quantum_kernel, x_train, y_train, x_test, y_test):
    """
    Evaluate QSVM performance on train and test sets.
    """
    # Training performance
    y_train_pred = model.predict(quantum_kernel.evaluate(x_vec=x_train, y_vec=x_train))
    train_acc = accuracy_score(y_train, y_train_pred)

    # Testing performance
    y_test_pred = model.predict(quantum_kernel.evaluate(x_vec=x_test, y_vec=x_train))
    test_acc = accuracy_score(y_test, y_test_pred)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_cm": confusion_matrix(y_train, y_train_pred),
        "test_cm": confusion_matrix(y_test, y_test_pred),
    }



def classical_svm(x_train, y_train, x_test, y_test):
    """
    Train a classical linear SVM as baseline.
    """
    model = SVC(kernel="linear")
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    return {
        "train_acc": accuracy_score(y_train, y_train_pred),
        "test_acc": accuracy_score(y_test, y_test_pred),
        "train_cm": confusion_matrix(y_train, y_train_pred),
        "test_cm": confusion_matrix(y_test, y_test_pred),
    }
