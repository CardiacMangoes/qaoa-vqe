import numpy as np
from qiskit.opflow import I, X, Z, Y

from qaoa import *


class QuantumModel:
    models = ['schwinger', 'ising']
    reprs = {'schwinger': ['__general__', 'standard', 'dimensionless'],
             'ising': ['__general__']}

    def __init__(self, model: str, rep = '__general__'):

        self.model = model.lower()
        self.rep = rep.lower()

        assert self.model in QuantumModel.models
        assert self.rep in QuantumModel.reprs[self.model]

    def make(self, params) -> QAOAHamiltonian:
        if self.model == 'schwinger':
            if self.rep == 'standard':
                params.update({'H_E': (params['g'] ** 2) * params['a'] / 2,
                               'H_M': params['m'] / 2,
                               'H_I': 1 / (4 * params['a'])})
            elif self.rep == 'dimensionless':
                params.update({'H_E': 1,
                               'H_M': np.sqrt(params['x']) / 2 * params['m/g'],
                               'H_I': params['x'] / 2})
            return self.make_schwinger(params)
        if self.model == 'ising':
            return self.make_ising(params)

    def make_ising(self, params) -> QAOAHamiltonian:
        num_qubits = params['N']
        cz = params['cz']

        target_coupling = CouplingTensor(num_qubits)
        mixer_coupling = CouplingTensor(num_qubits)

        for i in range(num_qubits):
            target_coupling[i:Z, (i + 1) % num_qubits: Z] += cz
            mixer_coupling[i:X] += 1

        return QAOAHamiltonian(target_coupling, mixer_coupling)

    def make_schwinger(self, params) -> QAOAHamiltonian:
        num_qubits = params['N']
        H_E, H_M, H_I = params['H_E'], params['H_M'], params['H_I']
        theta, mixer_type = params['theta'], params['mixer_type']

        assert num_qubits >= 2, "N has to be greater than or equal to 2"
        assert num_qubits <= 22, "N can be no greater than 22"
        assert num_qubits % 2 == 0, "N must be even"
        assert mixer_type in {'X', 'Y', 'XY'}, "mixer_type must be 'X', 'Y', or 'XY'"


        driver_coupling = CouplingTensor(num_qubits)
        mixer_coupling = CouplingTensor(num_qubits)
        target_coupling = CouplingTensor(num_qubits)
        
        # Constant
        driver_coupling[0:I] = (1 / 4) * ((num_qubits - 1) * (theta / np.pi) ** 2 + np.ceil((num_qubits - 1) / 2.0) *
                                          (1 + 2 * (theta / np.pi)) +
                                          ((num_qubits * (num_qubits - 1)) / 2)) * H_E

        # H_E
        for j in range(num_qubits - 1):
            for k in range(j):
                driver_coupling[j:Z, k:Z] += (num_qubits - j - 1) / 2.0 * H_E

        for j in range(num_qubits):
            driver_coupling[j:Z] = (1 / 2.0) * (
                    (theta / np.pi) * (num_qubits - j - 1) +
                    (np.ceil((num_qubits - j - 1) / 2) - ((num_qubits * j) % 2))) * H_E

        # H_M
        for i in range(num_qubits):
            driver_coupling[i:Z] += ((-1) ** i) * H_M

        target_coupling = driver_coupling.copy()

        # H_I
        for i in range(num_qubits - 1):
            target_coupling[i:X, (i + 1):X] = H_I
            target_coupling[i:Y, (i + 1):Y] = H_I

        if mixer_type in {'XY'}:
            for i in range(num_qubits - 1):
                mixer_coupling[i:X, (i + 1):X] = H_I
                mixer_coupling[i:Y, (i + 1):Y] = H_I
        elif mixer_type in {'X'}:
            for i in range(num_qubits):
                mixer_coupling[i:X] = H_I
        elif mixer_type in {'Y'}:
            for i in range(num_qubits):
                mixer_coupling[i:Y] = H_I

        return QAOAHamiltonian(driver_coupling, mixer_coupling, target_coupling)
