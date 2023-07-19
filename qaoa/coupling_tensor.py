import numpy as np
from qiskit.opflow import PauliOp, I, X, Y, Z

paulis = [I, X, Y, Z]


class CouplingTensor:
    def __init__(self, n: int, tensor={}) -> None:
        self.n = n
        self.tensor = tensor.copy()

    def keys(self):
        return self.tensor.keys()

    def values(self):
        return self.tensor.values()

    @staticmethod
    def key_type(key):
        if type(key) is slice:
            if type(key.stop) is PauliOp:
                return "single pauli"
        elif type(key) is tuple:
            k = key[0]
            if type(k) is slice:
                if type(k.stop) is PauliOp:
                    return "multi pauli"
            elif type(k) in {int, np.int32, np.int64}:
                return "coord"
        raise TypeError("Not pauli or coord")

    def get_coupling(self, key):
        coupling = np.zeros([self.n], dtype=int)
        ktype = CouplingTensor.key_type(key)
        if ktype == "single pauli":
            coupling[key.start] = paulis.index(key.stop)
        elif ktype == "multi pauli":
            for k in key:
                coupling[k.start] = paulis.index(k.stop)
        elif ktype == "coord":
            coupling = key[::-1]

        # reverse order to fix ordering of numpy array
        coupling = tuple(coupling[::-1])
        return coupling

    def to_pauli_op(self):
        result = None
        for ndx in self.tensor.keys():
            c = self.tensor[ndx]
            if c != 0:
                term = None
                for i in ndx:
                    term = term ^ paulis[i] if term else paulis[i]
                term *= c
                result = result + term if result else term
        return result

    def normalized(self):
        """
        Does not mutate self
        """

        min_coeff = min([abs(v) for v in self.tensor.values()])

        tensor = self.tensor.copy()
        tensor.update((x, y / min_coeff) for x, y in tensor.items())

        return CouplingTensor(self.n, tensor)

    def copy(self):
        return CouplingTensor(self.n, self.tensor.copy())

    def update(self, new_coupling):
        assert self.n == new_coupling.num_qubits
        self.tensor.update(new_coupling)

    def __setitem__(self, key, value: complex) -> None:
        coupling = self.get_coupling(key)
        self.tensor[coupling] = value
        if value == 0:
            del self.tensor[coupling]

    def __getitem__(self, key) -> complex:
        coupling = self.get_coupling(key)
        return self.tensor.get(coupling, 0)
    
    def __len__(self):
        return len(self.tensor)

    @property
    def __str__(self) -> str:
        return self.tensor.__str__

    @property
    def __repr__(self):
        return self.__str__
