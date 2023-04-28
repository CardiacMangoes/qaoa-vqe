import numpy as np
import plotly.graph_objects as go
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import ADAM, CG, COBYLA, L_BFGS_B, GradientDescent, NELDER_MEAD, NFT, POWELL, SLSQP, \
    SPSA, TNC
import scipy

import itertools
import time

from qaoa import QAOAHamiltonian, Database, EnergyLandscape, QAOA_VQE


class FindCurves:

    def __init__(self, hamiltonian: QAOAHamiltonian):
        self.hamiltonian = hamiltonian
        self.hash_ = hamiltonian.hash()
        self.p = 0

        min_gamma_coeff = self.hamiltonian.target_min_coeff
        min_beta_coeff = self.hamiltonian.mixer_min_coeff

        self.max_gamma = (2 / min_gamma_coeff) * np.pi
        self.max_beta = (2 / min_beta_coeff) * np.pi

        self.optimizers = [COBYLA(maxiter=5000), SLSQP(maxiter=5000), L_BFGS_B(maxiter=5000)]

    def find_next_curve(self):
        self.p += 1
        print(f'Finding curves for p = {self.p}')
        if self.p == 1:
            landscape = EnergyLandscape(self.qaoa_vqe)
            landscape.render_landscape()

        elif self.p == 2:
            min_points = Database().open(self.hash_).get('EnergyLandscape').get('min_points')
            initial_points = self.find_initial_points(min_points.get('gamma'), min_points.get('beta'))
            result = QAOA_VQE(self.hamiltonian, self.p).run_vqe(initial_points, self.optimizers)
            Database().save(self.hash_, ['curves', self.p], result)

    def find_initial_points(self, prior_gamma, prior_beta):
        initial_points = np.array([])
        for i, (gamma, beta) in enumerate(zip(prior_gamma, prior_beta)):
            if self.p == 1:
                for delta in [0.002, 0.02, 0.2]:
                    delta_gamma = self.max_gamma * delta
                    delta_beta = self.max_beta * delta

                    combos = list(itertools.product([beta[0] + delta_beta, beta[0] - delta_beta],
                                                    [beta[0] + delta_beta, beta[0] - delta_beta],
                                                    [gamma[0] + delta_gamma, gamma[0] - delta_gamma],
                                                    [gamma[0] + delta_gamma, gamma[0] - delta_gamma]))
                    for c in combos:
                        initial_points.append(c)
            else:
                s = np.arange(self.p - 1) / (self.p - 2)
                ind_var = np.arange(self.p) / (self.p - 1)
                for q in range(1, self.p - 1):
                    beta_coeffs = np.polyfit(s, beta, q)
                    beta_poly = np.poly1d(beta_coeffs)

                    gamma_coeffs = np.polyfit(s, gamma, q)
                    gamma_poly = np.poly1d(gamma_coeffs)

                    initial_points.append(np.append(beta_poly(ind_var), gamma_poly(ind_var)))

                beta_interp = scipy.interpolate.interp1d(s, beta, fill_value='extrapolate')
                gamma_interp = scipy.interpolate.interp1d(s, gamma, fill_value='extrapolate')
                initial_points.append(np.append(beta_interp(ind_var), gamma_interp(ind_var)))
        return initial_points

    def filter_data(optimal_params, optimal_values, cutoff):
        param_len = optimal_params.shape[1]
        unfiltered = set(range(len(optimal_params)))
        whitelist = set(range(len(optimal_params)))
        for i in range(len(optimal_params)):
            if i in whitelist:
                whitelist.remove(i)
                if optimal_values[i] < cutoff:
                    for j in whitelist:
                        if i in unfiltered and j in unfiltered:
                            if sum(abs(optimal_params[i] - optimal_params[j])) < 2e-1:
                                unfiltered.remove(j)
                else:
                    if i in unfiltered:
                        unfiltered.remove(i)

        take_this = np.array(list(unfiltered))
        param_take = np.stack([(take_this * param_len) + i for i in range(param_len)]).T
        filtered_values = np.take(optimal_values, take_this)
        filtered_params = np.take(optimal_params, param_take)
        return filtered_params, filtered_values

    def result_eval(target, mixer, filtered_params, filtered_values):
        npme = NumPyMinimumEigensolver()
        max_result = npme.compute_minimum_eigenvalue(operator=-1 * (target + mixer))
        min_result = npme.compute_minimum_eigenvalue(operator=target + mixer)

        max_energy = max_result.eigenvalue.real * -1
        min_energy = min_result.eigenvalue.real
        energy_range = abs(max_energy - min_energy)

        print(f'Maximum Energy: {max_energy:.5f}')
        print(f'Minimum Energy: {min_energy:.5f}')
        print(f'Energy Range: {energy_range:.5f}')

        ndx, minima = find_lowest(filtered_values)

        error = lambda val: abs(val - min_energy) / energy_range * 100
        min_error = error(minima)
        # print the parameters
        print(f"\nParams: {filtered_params[ndx]}")
        print(f"Energy: {minima}")
        print(f"Error: {min_error:.5f}%")

    def find_lowest(arr):
        min_ndx = 0
        minima = np.infty
        for i, v in enumerate(arr):
            if v < minima:
                min_ndx = i
                minima = v
        return min_ndx, minima

    def make_converge(result):
        names = result.keys()
        converge_cnts = np.concatenate([result[name]['converge_cnts'] for name in names])
        converge_params = np.concatenate([result[name]['converge_params'] for name in names])
        converge_vals = np.concatenate([result[name]['converge_vals'] for name in names])
        return converge_cnts, converge_params, converge_vals

    def make_optimal(converge_params, converge_vals):
        param_len = converge_params[0].shape[1]
        optimal_params = np.zeros([len(converge_params), param_len], dtype=object)
        optimal_values = np.zeros([len(converge_vals)], dtype=object)
        for i, z in enumerate(zip(converge_params, converge_vals)):
            params, vals = z[0], z[1]
            ndx, val = find_lowest(vals)
            optimal_params[i] = params[ndx]
            optimal_values[i] = val
        return optimal_params, optimal_values

    def graph_curves(parameters, values):
        fig = go.Figure()
        p = len(parameters[0]) // 2
        minima = np.min(values)
        for i, point in enumerate(parameters):
            param_len = len(point) // 2
            beta = point[:param_len]
            gamma = point[param_len:]
            if values[i] == minima:
                fig.add_trace(go.Scatter(x=np.arange(p) / (p - 1), y=beta, name='beta', marker={'color': 'darkred'}))
                fig.add_trace(go.Scatter(x=np.arange(p) / (p - 1), y=gamma, name='gamma', marker={'color': 'darkblue'}))
            else:
                fig.add_trace(go.Scatter(x=np.arange(p) / (p - 1), y=beta, name='beta', marker={'color': 'red'}))
                fig.add_trace(go.Scatter(x=np.arange(p) / (p - 1), y=gamma, name='gamma', marker={'color': 'blue'}))

        fig.update_layout(title=f'p = {p} Gamma and Beta Curves',
                          xaxis_title='Step (i-1)/(p-1)',
                          yaxis_title='Gamma(Blue)/Beta(Red)',
                          autosize=False,
                          width=1500, height=1000, )
        fig.show()

