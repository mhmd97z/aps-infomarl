import os
import sys
import yaml
import torch
import numpy as np
from torch_geometric.data import HeteroData
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../envs/aps/lib")))
from gnn_olp.gnn import FastGNNLinearPrecodingLightning
from aps_utils import opti_OLP, clip_abs, get_adj, sinr_from_A

feas_sinr_tol, feas_power_tol = 1e-3, 1e-6

class PowerControl:
    def __init__(self, conf):
        self.conf = conf
        self.tpdv = dict(
            device=self.conf.device_sim,
            dtype=self.conf.float_dtype_sim)

    def get_power_coef(self, G, rho_d):
        raise NotImplementedError("Subclasses must implement this method")

    def calcualte_sinr(self, G, rho_d, P):
        recv_power = G.T @ P # row i, col j: recv power at ue i, intended for ue j
        intened_power = torch.diag(recv_power)
        interfernce_power = recv_power.fill_diagonal_(0)
        numerator = rho_d * torch.abs(intened_power)**2
        denominator = 1 + rho_d * torch.sum(torch.abs(interfernce_power)**2, axis=1)
        sinr = numerator / denominator
        # to avoid -inf values:
        sinr[sinr == 0] = 1e-20
        if self.conf.if_sinr_in_db:
            return 10*torch.log10(sinr)
        else:
            return sinr

    def get_transmission_power(self, allocated_power):
        return allocated_power.abs() ** 2 \
            * self.conf.ap_radiation_power / self.conf.signal_transmission_efficiency

    def get_ap_circuit_power(self, mask):
        return torch.sum(mask, dim=1).sign() * self.conf.ap_constant_power_consumption * self.conf.ap_radiation_power

    def get_optimal_sinr(self, G, rho_d, mask=None):
        low, up, eps = 0, 10**6, 1e-6
        M, K = G.shape
        G = G.cpu().numpy()
        rho_d = rho_d.cpu().numpy()

        G_inv = np.linalg.inv((np.conjugate(G).T).dot(G))
        G_dague = np.conjugate(G).dot(G_inv.T)
        P_G = np.eye(M) - G_dague.dot(G.T)

        U_opt = np.zeros((M, K))
        A_opt = np.zeros((K, K))
        U_test = np.zeros((M, K))
        A_test = np.zeros((K, K))
        lowb = min(low, up)
        upb = max(low, up)
        ite = 0
        best_SINR = 0.0
        found_feasible_solution = False
        while abs(lowb-upb) > eps or not found_feasible_solution:
            ite += 1
            tSINR = (lowb+upb) / 2
            try:
                prob, A_test, U_test = opti_OLP(
                    tSINR, G_dague, P_G, rho_d, M, K, mask)
                is_feasible = False
                if prob.value is not None and prob.value < np.inf:
                    is_feasible = True
                    min_sinr = sinr_from_A(A_test.value, rho_d).min()
                    if min_sinr < tSINR * (1-feas_sinr_tol):
                        is_feasible = False
                    Delta = G_dague @ A_test.value + P_G @ U_test.value
                    max_power = np.linalg.norm(Delta, ord=2, axis=1).max()
                    if max_power > 1+feas_power_tol:
                        is_feasible = False
            except:
                is_feasible = False

            if is_feasible:
                lowb = tSINR
                A_opt, U_opt = A_test.value, U_test.value
                best_SINR = tSINR
                found_feasible_solution = True
            else:
                upb = tSINR

        Delta_opt = G_dague @ A_opt + P_G @ U_opt
        if mask is not None:
            Delta_opt = np.multiply(Delta_opt, mask)
        best_SINR = 10*np.log10(best_SINR)

        return best_SINR, Delta_opt


class OlpGnnPowerControl(PowerControl):
    def __init__(self, conf):
        super().__init__(conf)
        with open(self.conf.data_normalization_config, 'r') as config_file:
            self.normalization_dict = yaml.safe_load(config_file)
        self.graph_shape = (0, 0)
        self.load_model()

    def load_model(self):
        self.model = FastGNNLinearPrecodingLightning.load_from_checkpoint(
            self.conf.power_control_saved_model
        )
        self.model = self.model.eval()
        self.model = self.model.to(**self.tpdv)

    def graph_generation(self, n_aps, n_ues):
        same_ue_edges, same_ap_edges = get_adj(n_ues, n_aps, if_transpose=True)
        same_ue_edges = torch.tensor(same_ue_edges).t().contiguous().to(self.tpdv['device'])
        same_ap_edges = torch.tensor(same_ap_edges).t().contiguous().to(self.tpdv['device'])
        data = HeteroData()
        data['channel'].x = None
        data['channel', 'same_ue', 'channel'].edge_index = same_ue_edges
        data['channel', 'same_ap', 'channel'].edge_index = same_ap_edges
        return data

    def get_power_coef(self, G, rho_d, mask, return_graph=False):
        # pre-process
        number_of_aps, number_of_ues = G.shape
        if self.graph_shape != (number_of_aps, number_of_ues):
            self.graph_shape = (number_of_aps, number_of_ues)
            self.graph = self.graph_generation(number_of_aps, number_of_ues)
        G = clip_abs(G)
        G_T = G.T
        G_conj = torch.conj(G)
        G_inv = torch.inverse(G_conj.T @ G)
        G_dague = G_conj @ G_inv.T
        x = torch.reshape(G_T, (-1, 1))
        x1 = torch.reshape(G_dague.T, (-1, 1))
        x = torch.cat((torch.log2(torch.abs(x)), x.angle(),
                       torch.log2(torch.abs(x1)+1), x1.angle()), 1)
        x_mean = torch.tensor(self.normalization_dict['x_mean']).to(**self.tpdv)
        x_std = torch.tensor(self.normalization_dict['x_std']).to(**self.tpdv)
        x = (x - x_mean) / x_std
        x = torch.cat((x, mask.T.reshape(-1, 1)), dim=1)
        self.graph['channel'].x = x.to(**self.tpdv)
        self.graph['channel'].input_mean = torch.reshape(x_mean, (1, 4)).to(**self.tpdv)
        self.graph['channel'].input_std = torch.reshape(x_std, (1, 4)).to(**self.tpdv)
        self.graph['channel'].n_ues = number_of_ues
        self.graph['channel'].n_aps = number_of_aps
        self.graph['channel'].num_graph_node = number_of_ues * number_of_aps
        self.graph['channel'].rho_d = rho_d.to(**self.tpdv)

        with torch.no_grad():
            y, penultimate = self.model(self.graph)
            y, penultimate = y.to(**self.tpdv), penultimate.to(**self.tpdv)

        # post-process
        output_mean = torch.tensor(self.normalization_dict['y_mean']).to(**self.tpdv)
        output_std = torch.tensor(self.normalization_dict['y_std']).to(**self.tpdv)
        y = y * output_std + output_mean
        y = torch.polar(torch.pow(2, y[:, [0, 2, 4]]),
                        y[:, [1, 3, 5]])

        y1 = y[:, 0].view(number_of_ues, number_of_aps).T
        y2 = y[:, 1].view(number_of_ues, number_of_aps).T
        y3 = y[:, 2].view(number_of_ues, number_of_aps).T - 1e-20

        if self.tpdv['dtype'] == torch.float32:
            complex_type = torch.complex64
        if self.tpdv['dtype'] == torch.float64:
            complex_type = torch.complex128
        A1 = torch.matmul(G_T, y1).real.to(complex_type)
        A1 = torch.diag(torch.diag(A1))
        y1 = torch.matmul(G_dague, A1)
        A2 = torch.matmul(G_T, y2)
        y2 = torch.matmul(G_dague, A2 - torch.diag(torch.diag(A2)))
        power_coef = y1 + y2 + y3

        # normalize the allocated power to a max of one
        power = power_coef.clone()
        power = torch.linalg.norm(power, dim=1, keepdim=True)
        power_violated_index = power > 1
        power_ok_index = ~power_violated_index
        scaling_power = power_violated_index*power + power_ok_index
        scaling_power = scaling_power.expand(-1, number_of_ues)
        scaling_power = scaling_power.view(number_of_aps, number_of_ues)
        power_coef /= scaling_power

        if return_graph:
            return power_coef, penultimate.view(-1, number_of_aps * number_of_ues), self.graph.clone()
        else:
            return power_coef, penultimate.view(-1, number_of_aps * number_of_ues)


class MrtPowerControl(PowerControl):
    def __init__(self, conf):
        super().__init__(conf)

    def get_power_coef(self, G, mask):
        ap_connected_users_repeated = mask.sum(dim=1, keepdim=True).repeat(1, G.shape[1])
        power_budget = torch.ones_like(G).to(**self.tpdv) \
            * torch.sqrt(torch.tensor(1 / ap_connected_users_repeated)).to(**self.tpdv)
        power_coef = torch.conj(G) / torch.abs(G) * power_budget

        return power_coef
