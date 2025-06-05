import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from onpolicy.algorithms.fmat.utils.util import check, init
from onpolicy.algorithms.fmat.utils.transformer_act import discrete_parallel_act_infer
from onpolicy.algorithms.fmat.utils.transformer_act import discrete_parallel_act_train
from onpolicy.algorithms.fmat.utils.aps_generate_factor import get_factors, create_masks


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0.0), gain=gain)


class SelfAttention(nn.Module):

    # def __init__(self, n_embd, n_head, n_agent, masked=False):
    def __init__(self, n_embd, n_head, n_agent, masked=False, masked_factor=False, masked_node=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked_factor = masked_factor
        self.masked_node = masked_node
        self.masked = masked
        self.n_head = n_head
        self.n_agent = n_agent
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))

        self.att_bp = None

    def forward(self, key, value, query, factor_mask=None, node_mask=None):
        B, L, D = query.size()
        
        # q, k, v: (batch, n_agent+n_factor, n_embd)
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        n_agent = self.n_agent
        n_factor = L - n_agent


        # pdb.set_trace()
        # update factors
        # factor_mask: (n_factor, num_agent_related = 4), one factor has 4 related agent
        if self.masked_factor and factor_mask is not None:
            same_ue_factor_count = factor_mask[0].shape[0]
            # same_ap_factor_count = factor_mask[1].shape[0]

            # k, v: (batch, n_agent, n_embd)
            k = k[:,0:n_agent,:]
            v_keep = v[:,0:n_agent,:]
            v = v_keep

            ## same_ue factors
            q_same_ue = q[:,n_agent:n_agent+same_ue_factor_count,:].unsqueeze(-2)
            k_same_ue = k[:,factor_mask[0],:]
            v_same_ue = v[:,factor_mask[0],:]

            score = (q_same_ue @ k_same_ue.transpose(-2, -1)) * (1.0 / math.sqrt(k_same_ue.size(-1)))
            score = F.softmax(score, dim=-1)
            
            # y: (batch, n_factor, 1, n_embd)
            y_same_ue = score @ v_same_ue
            y_same_ue = y_same_ue.squeeze(-2).contiguous()

            ## same_ap factors
            q_same_ap = q[:,n_agent+same_ue_factor_count:,:].unsqueeze(-2)
            k_same_ap = k[:,factor_mask[1],:]
            v_same_ap = v[:,factor_mask[1],:]
            
            score = (q_same_ap @ k_same_ap.transpose(-2, -1)) * (1.0 / math.sqrt(k_same_ap.size(-1)))
            score = F.softmax(score, dim=-1)
            
            # y: (batch, n_factor, 1, n_embd)
            y_same_ap = score @ v_same_ap
            y_same_ap = y_same_ap.squeeze(-2).contiguous()
            
            # y: (batch, n_agent + n_factor, n_embd)
            y = torch.cat((v_keep, y_same_ue, y_same_ap), dim=-2)

        # update nodes   
        if self.masked_node and node_mask is not None: 
            # q: (batch, n_agent, n_embd)
            q = q[:,0:n_agent,:]
            # k, v: (batch, n_factor, n_embd)
            k = k[:,n_agent:,:]
            v_keep = v[:,n_agent:,:]
            v = v_keep

            # add dummy factor to make each node has same number of related factors
            # k, v: (batch, (r+1)*(r+1), n_embd)
            # k = pad_factor(k)
            # v = pad_factor(v)

            # use mask to gather related factors
            k = k[:,node_mask,:]
            v = v[:,node_mask,:]

            # q: (batch, n_agent, 1, n_embd)
            q = q.unsqueeze(-2)

            # score: (batch, n_agent, 1, num_factor_related)
            score = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            score = F.softmax(score, dim=-1)

            y = score @ v

            # y: (batch, n_agent, n_embd)
            y = y.squeeze(-2).contiguous()

            # y: (batch, n_agent + n_factor, n_embd)
            y = torch.cat((y, v_keep), dim=-2)

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.ln3 = nn.LayerNorm(n_embd, eps=1e-5)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked_factor=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked_node=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )


    def forward(self, x, factor_mask, node_mask):
        
        x = self.ln1(x + self.attn1(key=x, value=x, query=x, factor_mask=factor_mask))
        x = self.ln2(x + self.attn2(key=x, value=x, query=x, node_mask=node_mask))
        x = self.ln3(x + self.mlp(x))

        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.ln3 = nn.LayerNorm(n_embd, eps=1e-5)
        self.ln4 = nn.LayerNorm(n_embd, eps=1e-5)
        self.ln5 = nn.LayerNorm(n_embd, eps=1e-5)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked_factor=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked_node=True)
        self.attn3 = SelfAttention(n_embd, n_head, n_agent, masked_factor=True)
        self.attn4 = SelfAttention(n_embd, n_head, n_agent, masked_node=True)    
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )



    def forward(self, x, rep_enc, factor_mask, node_mask):
        #x is action: (batch, n_agent+n_factor, emb_dim)
        #rep_enc is observation representation got from encoder: (batch, n_agent+n_factor, emb_dim)
        x = self.ln1(x + self.attn1(key=x, value=x, query=x, factor_mask=factor_mask))
        x = self.ln2(x + self.attn2(key=x, value=x, query=x, node_mask=node_mask))
        x = self.ln3(rep_enc + self.attn3(key=x, value=x, query=rep_enc, factor_mask=factor_mask))
        x = self.ln4(rep_enc + self.attn4(key=x, value=x, query=rep_enc, node_mask=node_mask))
        x = self.ln5(x + self.mlp(x)) 

        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim, eps=1e-5),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim, eps=1e-5),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd, eps=1e-5)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd, eps=1e-5),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs, factor_mask, node_mask):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings
        # pdb.set_trace()

        x = self.ln(x)
        
        for block in self.blocks:
            x = block(x, factor_mask, node_mask)


        #x: (batch, n_agent+n_factor, emb_dim)
        rep = x
        
        #v: (batch, n_agent, 1)
        v_loc = self.head(x[:,0:self.n_agent,:])


        
        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        self.n_agent = n_agent
     

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim, eps=1e-5),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd, eps=1e-5),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd, eps=1e-5),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                raise
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim, eps=1e-5),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd, eps=1e-5),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd, eps=1e-5),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd, bias=False), activate=True),
                                                    nn.GELU())
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim, eps=1e-5),
                                             init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
            self.ln = nn.LayerNorm(n_embd, eps=1e-5)
         
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd, eps=1e-5),
                                      init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action_rep, obs_rep, obs, factor_mask, node_mask):
        # action: (batch, n_agent+n_factor, action_dim), action_dim = action_dim (2) + 1 
        # obs_rep: (batch, n_agent+n_factor, n_embd)
        # obs: (batch, n_agent, n_embd)
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
        
            x = self.ln(action_rep)
            for block in self.blocks:
                x = block(x, obs_rep, factor_mask, node_mask)
        
            logit = self.head(x)
        
        logit = logit[:, 0:self.n_agent,:]

        #logit: (batch, n_agent+n_factor, n_embd)  
        return logit


class MultiAgentTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_ap, n_ue, n_embd, n_head, n_block, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.tpdv_mask = dict(dtype=torch.long, device=device)
        self.action_type = action_type
        self.device = device

        # state unused
        state_dim = 37

        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd, eps=1e-5),
                                      init_(nn.Linear(n_embd, n_embd)))

        r = int(math.sqrt(n_agent))    
        self.factor_mask, self.node_mask = create_masks(n_ap, n_ue)
        self.factor_mask[0] = check(self.factor_mask[0]).to(**self.tpdv_mask)
        self.factor_mask[1] = check(self.factor_mask[1]).to(**self.tpdv_mask)
        self.node_mask = check(self.node_mask).to(**self.tpdv_mask)

        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)
          

        # put on cuda
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        #compute factors, obs: (bacth, n_agent+n_factor, obs_dim)
        obs_fac= get_factors(obs, self.factor_mask)
        # action = get_factors(action, self.factor_mask)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            available_actions = get_factors(available_actions, self.factor_mask)

        batch_size = obs.shape[0]  

        v_loc, obs_rep = self.encoder(state, obs_fac,self.factor_mask, self.node_mask)

        if self.action_type == 'Discrete':
            action_log, entropy = discrete_parallel_act_train(self.head, self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.factor_mask, self.node_mask, self.tpdv, available_actions)

        return action_log, v_loc, entropy


    # Inference Phase
    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
   
        obs_fac= get_factors(obs, self.factor_mask)
       
        if available_actions is not None:
            # pdb.set_trace()
            available_actions = check(available_actions).to(**self.tpdv)
            available_actions = get_factors(available_actions, self.factor_mask)
        

        batch_size = obs.shape[0]


        v_loc, obs_rep = self.encoder(state, obs_fac, self.factor_mask, self.node_mask)
 
        if self.action_type == "Discrete":
            
            # output_action, ouput_action_log: (batch, n_agent+n_factor, 1)
            # Inference phase
            output_action, output_action_log = discrete_parallel_act_infer(self.head, self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.factor_mask, self.node_mask, self.tpdv,
                                                                           available_actions, deterministic)
           
        
        # else:
        #     output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
        #                                                                      self.n_agent, self.action_dim, self.tpdv,
        #                                                                      deterministic)
              

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)


        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)


        obs= get_factors(obs, self.factor_mask)

        # v_tot: (batch, n_agent, 1)
        # obs_rep: (batch, n_agent + n_factor, emb_dim)
        v_tot, obs_rep = self.encoder(state, obs, self.factor_mask, self.node_mask)
        return v_tot

