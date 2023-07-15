import numpy as np
from qiskit.opflow import PauliOp, PauliSumOp, I, X, Z, Y
import scipy

from qaoa import CouplingTensor
from utils import hash_dict
from rich import print

paulis = [I, X, Y, Z]


class QAOAHamiltonian:
    def __init__(self, 
                 driver_coupling: CouplingTensor, 
                 mixer_coupling: CouplingTensor,
                 target_coupling: CouplingTensor) -> None:
        """
        Object that stores the hamiltonians of a QAOA circuit. 
        We normalize the range of the Driver and Mixer to be [0, 2pi]

        Args:
            driver_coupling (CouplingTensor): Driver Hamiltonian (gamma)
            mixer_coupling (CouplingTensor): Mixer Hamiltonian (beta)
        """
        assert driver_coupling.num_qubits == mixer_coupling.num_qubits, "Driver and Mixer qubits must match"
        assert driver_coupling.num_qubits == target_coupling.num_qubits, "Target qubits must match Driver and Mixer"

        self.num_qubits = driver_coupling.num_qubits

        self.driver_coupling = driver_coupling.copy()
        self.mixer_coupling = mixer_coupling.copy()
        self.target_coupling = target_coupling.copy()

        self.driver_coupling = self.driver_coupling.normalized()
        self.mixer_coupling = self.mixer_coupling.normalized()

        self.driver = self.driver_coupling.to_pauli_op()
        self.mixer = self.mixer_coupling.to_pauli_op()
        self.target = self.target_coupling.to_pauli_op()

        self.eigenvalues = None
        self.eigenvectors = None

    def eigen(self, k = None):
        """Finds the eigenvalues and eigenvectors of the Target Hamiltonian

        Args:
            k (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if k is None:
            k = (2 ** self.num_qubits) - 2

        if self.num_qubits <= 5:
            matrix_ham = self.target.to_matrix()
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(matrix_ham)
        else:
            sparse_matrix_ham = self.target.to_spmatrix()
            self.eigenvalues, self.eigenvectors = scipy.sparse.linalg.eigsh(sparse_matrix_ham, which='SA', k=k)

        return self.eigenvalues[:k], self.eigenvectors.T[:k]

    def eigen_values(self, k=None):
        if k is None:
            eval_matrix = self.target.to_matrix(massive=True)
            return np.sort(np.linalg.eigvalsh(eval_matrix))
        else:
            eigenvalues =  self.eigen(k)[0]
            if k == 1:
                return eigenvalues[0]
            return eigenvalues
        

    def eigen_vectors(self, k=None):
        eigenvectors = self.eigen(k)[1]
        if k == 1:
            return eigenvectors[0]
        return eigenvectors

    def hash(self):
        coupling = self.driver_coupling.copy()
        coupling.update(self.mixer_coupling)
        return hash_dict(coupling.tensor)

    def __str__(self):
        return f"driver:\n{self.driver}\nmixer:\n{self.mixer}\ntarget:\n{self.target}"

    def __repr__(self):
        return self.__str__()
