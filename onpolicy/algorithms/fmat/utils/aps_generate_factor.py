import torch

def get_factors(obs, factor_mask):
    # obs: (batch, num_agent, D)
    # output: (batch, num+agent + n_factor, D)
    same_ue_factor_mask, same_ap_factor_mask = factor_mask

    same_ue_factor_obs = obs[:,same_ue_factor_mask.long(),:].float()
    same_ue_factor_reps = torch.mean(same_ue_factor_obs, dim=2)

    same_ap_factor_obs = obs[:,same_ap_factor_mask.long(),:].float()
    same_ap_factor_reps = torch.mean(same_ap_factor_obs, dim=2)
    
    obs_fac = torch.cat((obs, same_ue_factor_reps, same_ap_factor_reps), dim=1)

    return obs_fac

def create_masks(n_ap, n_ue):
    # two types of factors:
    #   ap based: each row of G(or P) matrix
    #   ue based: each colomn of G(or P) matrix

    ## same_ue factor
    # factor mask: (n_factor, factor_size) == (n_ue, n_ap)
    same_ue_factor_mask_list = [[] for _ in range(n_ue)]
    for cntr in range(n_ap*n_ue):
        same_ue_factor_mask_list[int(cntr%n_ue)].append(cntr)
    same_ue_factor_mask = torch.tensor(same_ue_factor_mask_list)

    same_ap_factor_mask_list = [[] for _ in range(n_ap)]
    for cntr in range(n_ap*n_ue):
        same_ap_factor_mask_list[int(cntr/n_ue)].append(cntr)
    same_ap_factor_mask = torch.tensor(same_ap_factor_mask_list)

    # node mask: (n_agents, associated_factors) == (n_ue*n_ap, 2)
    node_mask_list = [[] for _ in range(n_ue*n_ap)]
    for cntr in range(n_ap*n_ue):
        node_mask_list[cntr].append(int(cntr%n_ue))
        node_mask_list[cntr].append(n_ue + int(cntr/n_ue))
    node_mask = torch.tensor(node_mask_list)
    
    return [same_ue_factor_mask, same_ap_factor_mask], node_mask
