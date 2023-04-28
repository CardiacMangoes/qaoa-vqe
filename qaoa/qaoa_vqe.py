import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import ADAM, CG, COBYLA, L_BFGS_B, GradientDescent, NELDER_MEAD, NFT, POWELL, SLSQP, \
    SPSA, TNC
from qiskit.circuit.library import QAOAAnsatz, EvolvedOperatorAnsatz
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.parametervector import ParameterVector

import time


class QAOA_VQE:
    def __init__(self, hamiltonian, p, backend=None):
        self.hamiltonian = hamiltonian
        self.p = p
        self.ansatz = self.make_ansatz()
        
        self.backend = backend or Aer.get_backend('unitary_simulator') # short-circuits if backend is not None
        self.vqe = self.make_vqe()
        
        self.grnd_energy = NumPyMinimumEigensolver(). \
            compute_minimum_eigenvalue(operator=self.hamiltonian.target).eigenvalue

    def make_ansatz(self):
        # ansatz = QAOAAnsatz(cost_operator=self.hamiltonian.driver,
        #                     mixer_operator=self.hamiltonian.mixer,
        #                     reps=self.p,
        #                     initial_state=None,
        #                     name='QAOA')

        initial_state = QuantumCircuit(self.hamiltonian.num_qubits)
        initial_state.h(range(self.hamiltonian.num_qubits))

        ansatz = EvolvedOperatorAnsatz(operators=[-self.hamiltonian.driver, -self.hamiltonian.mixer],
                                       reps=self.p,
                                       name='QAOA',
                                       initial_state=initial_state)

        # parameterize circuit
        betas = ParameterVector("β", self.p)
        gammas = ParameterVector("γ", self.p)

        reordered = []
        for rep in range(self.p):
            reordered.extend(gammas[rep: (rep + 1)])
            reordered.extend(betas[rep: (rep + 1)])

        ansatz.assign_parameters(dict(zip(ansatz.ordered_parameters, reordered)), inplace=True)
        return ansatz

    def make_vqe(self):
        vqe = VQE(ansatz=self.ansatz,
                  quantum_instance=QuantumInstance(backend=self.backend))
        return vqe

    def vqe_energy(self):
        energy_eval = self.vqe.get_energy_evaluation(self.hamiltonian.target)

        def calc_vqe_energy(*args):
            return energy_eval(args)

        return calc_vqe_energy

    def run_vqe(self, initial_points, optimizers, verbose=False):
        final = {}
        print(f"Optimizing on {len(initial_points) * len(optimizers)} points")

        start = time.time()
        for optimizer in optimizers:
            name = type(optimizer).__name__
            converge_cnts = np.zeros([len(initial_points)], dtype=object)
            converge_params = np.zeros([len(initial_points)], dtype=object)
            converge_vals = np.zeros([len(initial_points)], dtype=object)
            for count, point in enumerate(initial_points):
                counts = []
                params = []
                values = []

                def store_intermediate_result(eval_count, parameters, mean, std):
                    counts.append(eval_count)
                    params.append(parameters)
                    values.append(mean)

                self.vqe.optimizer = optimizer
                self.vqe.initial_point = point
                self.vqe.callback = store_intermediate_result

                result = self.vqe.compute_minimum_eigenvalue(operator=self.hamiltonian.target)
                converge_cnts[count] = np.asarray(counts)
                converge_params[count] = np.asarray(params)
                converge_vals[count] = np.asarray(values)
                if verbose:
                    print(f"{name} {count + 1}\t\t\t", end='\r')
                final[name] = {'converge_cnts': converge_cnts, 'converge_params': converge_params,
                               'converge_vals': converge_vals}
        print(f"Rendered In: {time.time() - start : .2f}s" + '\t' * 20)
        return final
