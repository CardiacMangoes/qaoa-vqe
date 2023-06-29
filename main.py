import numpy as np
import scipy
import sympy as sp

import multiprocessing
import plotly.graph_objects as go

from qaoa import *

import pickle
import time
import warnings
warnings.filterwarnings("ignore")

class ParametersNode:
    def __init__(self, 
                 parameters,
                 energy):
        
        self.parameters = parameters
        self.energy = energy

        self.id = tuple(np.round(parameters, 12))

        self.p = len(parameters) // 2

        self.gammas = parameters[::2]
        self.betas = parameters[1::2]

        self.layer = None

        self.prev = {}
        self.next = {}

class NodeLayer:
    def __init__(self):
        self.nodes = {}
        self.num_nodes = 0

        self.prev = None
        self.next = None

    def process_new_nodes(self, all_new_nodes):
        partition_size = len(all_new_nodes) // len(self.nodes)
        for i, node in enumerate(self.nodes.values()):
            new_nodes = all_new_nodes[partition_size * i: partition_size * (i + 1)]

            if self.next is None:
                self.next = NodeLayer()
                self.next.prev = self

            for new_node in new_nodes:
                if new_node.id in self.next.nodes.keys():
                    new_node = self.next.nodes[new_node.id]
                else:
                    self.next.add_node(new_node)

                node.next[new_node.id] = new_node
                new_node.prev[node.id] = node

    def add_node(self, new_node: ParametersNode):
        self.nodes[new_node.id] = new_node
        self.num_nodes = len(self.nodes)

def square(*args):
    return np.sum(np.array(args) ** 2)

def optimize(energy_fn, initial_parameters):
        next_p = len(initial_parameters) // 2

        next_min = scipy.optimize.minimize(
                                    fun = lambda x: energy_fn(*x), 
                                    x0 = initial_parameters % (2 * np.pi),
                                    method = "L-BFGS-B" ,
                                    bounds = [[0, 2 * np.pi] for _ in range(next_p * 2)],
                                    options = {"gtol" : 1e-2})
        
        next_node = ParametersNode(parameters = next_min.x,
                                   energy = next_min.fun)
        
        return next_node

def find_landscape_minimas(landscape, filter = True):
    less_right = landscape <= np.roll(landscape, 1 , 0)
    less_left = landscape <= np.roll(landscape, -1 , 0)
    less_gamma = np.logical_and(less_right, less_left)

    less_bottom = landscape <= np.roll(landscape, 1 , 1)
    less_top = landscape <= np.roll(landscape, -1 , 1)
    less_beta = np.logical_and(less_bottom, less_top)

    minimas = np.logical_and(less_gamma, less_beta)

    res = landscape.shape[0]
    s = np.linspace(0, 2 * np.pi, res + 1)[:-1]

    x, y = np.meshgrid([s], [s])
    min_x = x[minimas]
    min_y = y[minimas]
    min_energy = landscape[minimas]

    if filter:
        filtered = min_energy <= np.mean(min_energy)
        min_x = min_x[filtered]
        min_y = min_y[filtered]
        min_energy = min_energy[filtered]

    return np.vstack([min_x, min_y, min_energy])

def closeness_to_ground(val, ham):
    return 1 - (ham.eigen_values()[-1] - val) / (ham.eigen_values()[-1] - ham.eigen_values()[0])

def interpolate_params(parameters):
        p = len(parameters) // 2
        gammas, betas = parameters[::2], parameters[1::2]

        x_fit = np.linspace(0, 1, p)
        x_poly = np.linspace(0, 1, p + 1)

        gamma_interp = np.interp(x_poly, x_fit, gammas)
        beta_interp = np.interp(x_poly, x_fit, betas)
        new_parameters = np.array([val for pair in zip(gamma_interp, beta_interp) for val in pair])
        return new_parameters

def fit_params(parameters, order):
    p = len(parameters) // 2
    gammas, betas = parameters[::2], parameters[1::2]

    x_fit = np.linspace(0, 1, p)
    x_poly = np.linspace(0, 1, p + 1)

    gamma_fit = np.poly1d(np.polyfit(x_fit, gammas, order))
    beta_fit = np.poly1d(np.polyfit(x_fit, betas, order))

    new_parameters = np.array([val for pair in zip(gamma_fit(x_poly), beta_fit(x_poly)) for val in pair])
    return new_parameters

def save(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':

    verbose = True
    cpu_count = multiprocessing.cpu_count()
    
    start = time.time()

    """
    Schwinger
    """
    select = {'N': 4, #qubits
            'g' : 1,  #coupling
            'm' : 1,  #bare mass
            'a' : 1, #lattice spacing
            'theta' : 0, #topological term
            'mixer_type' : 'Y', # type of mixer {'X', 'Y', 'XY'}
            }

    model = QuantumModel('schwinger', 'Standard')
    ham = model.make(select)

    layers = [NodeLayer()]

    ### Find p = 1 minimas ###

    symqaoa = QAOA_Sympy(ham, p=1)

    if verbose: print(f"Making p = 1 Expectation {time.time() - start}")
    symqaoa.make()

    if verbose: print(f"Making p = 1 Landscape {time.time() - start}")
    sym_energies = symqaoa.energy_landscape(res=128)

    energy_min_x, energy_min_y, min_energies  = find_landscape_minimas(sym_energies)

    minimas = []
    avg_energy = 0
    for x, y in zip(energy_min_x, energy_min_y):
        minima = scipy.optimize.minimize(fun = lambda x: symqaoa.energy(*x), 
                                x0 = [x, y],
                                method = "SLSQP" ,
                                bounds = [[0, 2 * np.pi] for _ in range(2)],
                                tol = 1e-16
                                )
        avg_energy += minima.fun 
        minimas.append(minima)
        
    avg_energy /= len(minimas)

    # filter minimas by throwing out any greater than the average
    minimas = [minima for minima in minimas if minima.fun <= avg_energy]

    if verbose:print("\np = 1 energies:")
    for minima in minimas:
        if verbose: print(f"{closeness_to_ground(minima.fun, ham) * 100:.6}%")
        node = ParametersNode(minima.x, minima.fun)
        layers[0].add_node(node)
    if verbose:print("")

    ### Find p = 2 minimas ###
    
    def perturb_params(perturbation):
        def parameter_fn(parameters):
            new_parameters = np.hstack([parameters, parameters])
            return new_parameters + perturbation
        return parameter_fn
    
    initial_parameters = []

    magnitudes = np.linspace(0, np.pi, 5)[1:-1]

    for node in layers[0].nodes.values():
        for magnitude in magnitudes:
            perturbations = [0, magnitude, -magnitude]
            for pgamma in perturbations:
                for pbeta in perturbations:
                    initial_parameters.append(perturb_params(np.array([0, 0, pgamma, pbeta]))(node.parameters))

    n = len(initial_parameters)

    if verbose: print(f"Optimizing p = 2, n = {n}, time = {time.time() - start:.6}")

    numqaoa = QAOA_Numpy(ham)

    p = multiprocessing.Pool(cpu_count)
    all_new_nodes = p.starmap(optimize, zip([numqaoa.energy] * n, initial_parameters))
    p.close()

    node_parameters = []
    node_energies = []

    prior_min_energy = min([node.energy for node in layers[0].nodes.values()])

    for node in all_new_nodes:
        node_parameters.append(node.parameters)
        node_energies.append(node.energy)

    layers[0].process_new_nodes(all_new_nodes)

    layers.append(NodeLayer())
    
    if verbose: print("\np = 2 energies:")

    count = 0
    for node in layers[0].next.nodes.values():
        count += 1
        if node.energy <= prior_min_energy:
            # if verbose: print(f"{closeness_to_ground(node.energy, ham) * 100:.4}%")
            layers[1].add_node(node)

    print(count)

    new_min_energy = min([node.energy for node in layers[0].next.nodes.values()])
    print(f"{closeness_to_ground(new_min_energy, ham) * 100:.5}%")
    if verbose: print("")
    

    save("layers", layers)

    ### Find p >= 3 minimas ###
    
    # for i in range(1, 100):
    #     initial_parameters = []

    #     for node in layers[i].nodes.values():
    #         initial_parameters.append(interpolate_params(node.parameters))
    #         initial_parameters.append(fit_params(node.parameters, 1))
    #         initial_parameters.append(fit_params(node.parameters, 2))
    #         initial_parameters.append(fit_params(node.parameters, 3))

    #     n = len(initial_parameters)

    #     if verbose: print(f"Optimizing p = {i + 2}, n = {n}, time = {time.time() - start:.6}s")

    #     p = multiprocessing.Pool(cpu_count)
    #     all_new_nodes = p.starmap(optimize, zip([numqaoa.energy] * n, initial_parameters))
    #     p.close()

    #     layers[i].process_new_nodes(all_new_nodes)

    #     layers.append(NodeLayer())

    #     prior_min_energy = min([node.energy for node in layers[i].nodes.values()])

    #     if verbose: print(f"\np = {i + 2} energies:")
    #     for node in layers[i].next.nodes.values():
    #         if node.energy <= prior_min_energy:
    #             # if verbose: print(f"{closeness_to_ground(node.energy, ham) * 100:.4}%")
    #             layers[i + 1].add_node(node)
        

    #     new_min_energy = min([node.energy for node in layers[i].next.nodes.values()])
    #     print(f"{closeness_to_ground(new_min_energy, ham) * 100:.5}%")
    #     if verbose: print("")

    #     save("layers", layers)

    print(f"Total Time: {time.time() - start}")