from qiskit import QuantumCircuit, assemble, Aer
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.circuit.parametervector import ParameterVector
from qiskit.opflow import I, X, Y, Z, H, CircuitStateFn, MatrixEvolution, PauliTrotterEvolution, StateFn, PauliExpectation, CircuitSampler
from qiskit_aer import QasmSimulator, StatevectorSimulator

from functools import lru_cache
from multiprocessing import Pool
import numpy as np
import sympy as sp
import sympy.physics.quantum as spq
from scipy import sparse
from scipy.sparse.linalg import expm
from tqdm import tqdm

import time

import warnings
warnings.filterwarnings("ignore")


#####################
####### Sympy #######
#####################

def rationalize(v):
    return sp.Rational(v).limit_denominator(100000)


def to_pauli_op(coupling, paulis=[I, X, Y, Z]):
    result = None
    for ndx in coupling.tensor.keys():
        c = coupling.tensor[ndx]
        if c != 0:
            term = None
            for i in ndx:
                term = term ^ paulis[i] if term else paulis[i]
            term *= c
            result = result + term if result else term
    return result


def exp_diag_mat(matrix):
    size = matrix.shape[0]

    result = sp.SparseMatrix(sp.eye(size))
    for i in range(size):
        result[i, i] = sp.exp(matrix[i, i])
    return result.copy()


def exp_diag(coupling, var, paulis=[I, X, Y, Z]):
    """
    Exponentiates the diagonal elements of a matrix
    """
    diag = to_pauli_op(coupling, paulis).to_matrix()
    result = sp.SparseMatrix(diag) * -1j * var

    return exp_diag_mat(result)


def exp_(coupling, var):
    paulis_set = set([i for sub in coupling.keys() for i in sub])
    # print(to_pauli_op(coupling))

    if 1 not in paulis_set and 2 not in paulis_set:  # Diagonal
        return exp_diag(coupling, var)

    elif 1 in paulis_set and 2 not in paulis_set and 3 not in paulis_set:  # Just X
        v = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
        D = exp_diag(coupling, var, paulis=[I, Z, I, I])
        P = spq.TensorProduct(*[v] * coupling.num_qubits)

    elif 2 in paulis_set and 1 not in paulis_set and 3 not in paulis_set:  # Just Y
        v = sp.Matrix([[1, 1], [1j, -1j]]) / sp.sqrt(2)
        D = exp_diag(coupling, var, paulis=[I, I, Z, I])
        P = spq.TensorProduct(*[v] * coupling.num_qubits)

    else:  # No Structure
        A = to_pauli_op(coupling).to_matrix()
        D, P = np.linalg.eigh(A)

        tol = 1e-10
        D[abs(D) < tol] = 0
        P[abs(P) < tol] = 0

        D = sp.SparseMatrix(np.diag(D))
        P = sp.SparseMatrix(P).applyfunc(rationalize)

        D = exp_diag_mat(D * -1j * var)

    # print("multiplying components")
    result = P * D * P.adjoint()
    return result.copy()


class QAOA_Sympy:
    def __init__(self, hamiltonian, p, verbose=False):
        self.driver_coupling = hamiltonian.driver_coupling.copy()
        self.mixer_coupling = hamiltonian.mixer_coupling.copy()
        self.target = sp.SparseMatrix(hamiltonian.target.to_matrix())

        self.num_qubits = hamiltonian.num_qubits
        self.p = p

        self.gammas = sp.symbols([f"gamma_{i + 1}" for i in range(self.p)], real=True)
        self.betas = sp.symbols([f"beta_{i + 1}" for i in range(self.p)], real=True)
        self.parameters = [val for pair in zip(self.gammas, self.betas) for val in pair]

        self.energy_expr = None
        self.energy_fn = None
        self.wavefunction = None

        self.verbose = verbose

        self.grnd_energy = hamiltonian.eigen_values(k=1)
        self.grnd_state = sp.Matrix(hamiltonian.eigen_vectors(k=1))

        self.driver = None
        self.mixer = None

        self.wavefunction = self._initialize_wavefunction()


    def _initialize_wavefunction(self):
        if self.verbose: start = time.time()

        wavefunction = sp.SparseMatrix(np.ones(2 ** self.num_qubits)) / sp.sqrt(2 ** self.num_qubits)

        if self.verbose: print(f"Making Driver")
        self.driver = exp_(self.driver_coupling, self.gammas[0])
        if self.verbose: print(time.time() - start)

        if self.verbose: print(f"Making Mixer")
        self.mixer = exp_(self.mixer_coupling, self.betas[0])
        if self.verbose: print(time.time() - start)

        if self.verbose: print(f"Making wavefunction")
        for i in range(self.p):
            wavefunction = self.driver.subs({self.gammas[0]: self.gammas[i]}) * wavefunction
            wavefunction = self.mixer.subs({self.betas[0]: self.betas[i]}) * wavefunction
        if self.verbose: print(time.time() - start)

        return wavefunction

    def add_round(self):
        self.p += 1

        new_gamma = sp.symbols(f"gamma_{self.p}", real=True)
        new_beta = sp.symbols(f"beta_{self.p}", real=True)

        self.gammas.append(new_gamma)
        self.betas.append(new_beta)
        self.parameters.extend([new_gamma, new_beta])

        self.wavefunction = self.driver.subs({self.gammas[0]: new_gamma}) * self.wavefunction
        self.wavefunction = self.mixer.subs({self.betas[0]: new_beta}) * self.wavefunction

    def make(self):
        self.make_expectation()

    def make_expectation(self):
        if self.verbose: start = time.time()

        if self.verbose: print(f"Finding Result")
        result = self.wavefunction.adjoint() * self.target * self.wavefunction
        self.energy_expr = result[0]
        if self.verbose: print(time.time() - start)

        if self.verbose: print(f"Lambdifying")
        self.energy_fn = sp.lambdify(self.parameters, self.energy_expr, modules="numpy")
        if self.verbose: print(time.time() - start)

    def energy(self, *args):
        args = list(args) + (self.p * 2 - len(args)) * [0]
        return self.energy_fn(*args).real
    
    def energy_landscape(self, res):
        s = np.linspace(0, 2 * np.pi, res + 1)[:-1]
        return self.energy_fn(*np.meshgrid(*[s, s] * self.p)).real


######################
####### Numpy #######
######################


class QAOA_Numpy:
    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian

        self.num_qubits = hamiltonian.num_qubits

        self.plus = np.ones(2 ** self.num_qubits) / np.sqrt(2 ** self.num_qubits) # plus state
        self.driver = sparse.lil_matrix(self.hamiltonian.driver.to_matrix())
        self.mixer = sparse.lil_matrix(self.hamiltonian.mixer.to_matrix())
        self.target = sparse.lil_matrix(self.hamiltonian.target.to_matrix())

        self.grnd_energy = hamiltonian.eigen_values(k=1)
        self.grnd_state = hamiltonian.eigen_vectors(k=1)
     
    
    def energy(self, gammas, betas):

        assert len(gammas) == len(betas)

        wavefunction = self.plus.copy()

        for gamma, beta in zip(gammas, betas):
            driver = expm(self.driver * -1j * gamma)
            mixer = expm(self.mixer * -1j * beta)
            wavefunction = mixer @ (driver @ wavefunction)

        result = wavefunction.T.conjugate() @ self.target @ wavefunction
        return result.real
    
    def energy_landscape(self, res, p):
        s = np.linspace(0, 2 * np.pi, res + 1)[:-1]
        
        # create grid of gammas and betas
        dims = np.meshgrid(*[s] * (2 * p))
        coords = [dim.flatten() for dim in dims]
        parameters = np.array(coords).T.reshape(res ** (2*p), 2, p)

        # multiprocess inputs
        pool = Pool(processes = 12) 
        energy = pool.starmap(func=self.energy, iterable=tqdm(parameters))
        pool.close()

        landscape = np.reshape(np.array(energy), [res] * (2 * p))

        if p == 1:
            landscape = landscape.transpose(1, 0)
        if p == 2:
            landscape = landscape.transpose(1, 0, 2, 3)

        return landscape
    
    def state_ssd(self, gammas, betas):

        assert len(gammas) == len(betas)

        wavefunction = self.plus.copy()

        for gamma, beta in zip(gammas, betas):
            driver = expm(self.driver * -1j * gamma)
            mixer = expm(self.mixer * -1j * beta)

            wavefunction = mixer @ (driver @ wavefunction)

        real_diff = np.real(self.grnd_state) - np.real(wavefunction)
        imag_diff = np.imag(self.grnd_state) - np.imag(wavefunction)
        result = np.mean(real_diff ** 2 + imag_diff ** 2) 
        return result
    
    def state_ssd_landscape(self, res, p):
        s = np.linspace(0, 2 * np.pi, res + 1)[:-1]
        
        # create grid of gammas and betas
        dims = np.meshgrid(*[s] * (2 * p))
        coords = [dim.flatten() for dim in dims]
        parameters = np.array(coords).T.reshape(res ** (2*p), 2, p)

        # multiprocess inputs
        pool = Pool(processes = 12) 
        state_ssd = pool.starmap(func=self.state_ssd, iterable=tqdm(parameters))
        pool.close()

        landscape = np.reshape(np.array(state_ssd), [res] * (2 * p))

        if p == 1:
            landscape = landscape.transpose(1, 0)
        if p == 2:
            landscape = landscape.transpose(1, 0, 2, 3)

        return landscape



######################
####### Qiskit #######
######################


class QAOA_Qiskit:
    def __init__(self, hamiltonian, p):
        self.hamiltonian = hamiltonian
        self.p = p

        self.num_qubits = hamiltonian.num_qubits

        self.psi = np.ones(2 ** self.num_qubits) / np.sqrt(2 ** self.num_qubits) # plus state
        self.driver = self.hamiltonian.driver
        self.mixer = self.hamiltonian.mixer
        self.target = self.hamiltonian.target

        self.ansatz = self.construct_ansatz()

    def construct_ansatz(self):
        """
        Constructs a paramaterized QAOA ansatz given a QAOAHamiltonian for p number of rounds

        Returns:
            QuantumCircuit: Paramaterized QAOA Ansatz
        """
        ansatz = QuantumCircuit(self.num_qubits)

        # Create Parameters
        gammas = ParameterVector("γ", self.p) # driver
        betas = ParameterVector("β", self.p) # mixer

        # Initialize circuit to |+> state
        ansatz.h(range(self.num_qubits))

        # Apply e^{-i H_driver} e^{-i H_mixer} p number of timees
        for i in range(self.p):

            # Construct and apply driver unitary
            evolved_driver = (-1 * self.driver * gammas[i]).exp_i()
            evolved_driver = MatrixEvolution().convert(evolved_driver).reduce()
            driver_circuit = evolved_driver.to_circuit()
            ansatz = ansatz.compose(driver_circuit)

            # Construct and apply mixer unitary
            evolved_mixer = (-1 * self.mixer * betas[i]).exp_i()
            evolved_mixer = MatrixEvolution().convert(evolved_mixer).reduce()
            mixer_circuit = evolved_mixer.to_circuit()
            ansatz = ansatz.compose(mixer_circuit)

        return ansatz
    
    def expectation(self, wavefunction, type="approx"):
        """
        Evaluates the energy expectation value of an Wavefunction acting on an Observable

        Args:
            wavefunction_circ (QuantumCircuit): Wavefunction
            operator (PauliSumOp | PauliOp): Observable
            type (str): type of evaluation; 'exact' or 'approx'

        Returns:
            float: energy expectation value of an Wavefunction acting on an Observable
        """
        assert type in {"exact", "approx"}, "Type must be 'exact' or 'approx'"
        wavefunction_state = CircuitStateFn(wavefunction)

        if type == "exact":
            # <psi| operator |psi>
            eval_circuit = wavefunction_state.adjoint().compose(self.target).compose(wavefunction_state)
            result = eval_circuit.eval().real
        elif type == "approx":
            measurable_expression = StateFn(self.target, is_measurement=True).compose(wavefunction_state)
            expectation_function = PauliExpectation().convert(measurable_expression)

            backend = StatevectorSimulator()
            sampler = CircuitSampler(backend).convert(expectation_function)

            result = sampler.eval().real

        return result

    def energy(self, parameters):
        gammas, betas = parameters
        wavefunction = self.ansatz.assign_parameters(np.hstack([betas, gammas]))
        return self.expectation(wavefunction, type="approx")

    