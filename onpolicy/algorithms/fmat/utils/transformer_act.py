from torch.distributions import Categorical, Normal
from torch.nn import functional as F

def discrete_parallel_act_infer(head, decoder, obs_rep, obs, batch_size, n_agent, action_dim, factor_mask, node_mask,tpdv,
                                available_actions=None, deterministic=False):
    
    _, n_total, _ = obs_rep.size()
    init_action_rep = head(obs_rep)

    logit = decoder(init_action_rep, obs_rep, obs, factor_mask, node_mask)
 
    distri = Categorical(logits=logit)
    output_action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
    output_action_log = distri.log_prob(output_action)
  
    output_action = output_action.unsqueeze(-1)
    output_action_log = output_action_log.unsqueeze(-1)

    return output_action, output_action_log


def discrete_parallel_act_train(head, decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, factor_mask, node_mask, tpdv,
                          available_actions=None):
    _, n_total, _ = obs_rep.size()
    init_action_rep = head(obs_rep)
    logit = decoder(init_action_rep, obs_rep, obs, factor_mask, node_mask)

    distri = Categorical(logits=logit)

    output_action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)

    entropy = distri.entropy().unsqueeze(-1)
    
    return output_action_log, entropy
