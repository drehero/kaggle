import inspect
import math

import torch
import torch.nn.functional as F
import torch.nn as nn


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class InputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_projection = nn.Linear(1, config.hidden_size)
        self.xyz_projection = nn.Linear(3, config.hidden_size)
        #self.sensor_embd = nn.Embedding(5161, config.hidden_size)
        self.charge_projection = nn.Linear(1, config.hidden_size)
        self.aux_embd = nn.Embedding(3, config.hidden_size)
        self.pos_embd = nn.Embedding(config.max_len, config.hidden_size)
        
        self.layer_norm = LayerNorm(config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        self.device = config.device
        
    def forward(self, xyz, time, charge, aux):#, sensor_id):
        pos = torch.arange(0, xyz.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)
        x = self.xyz_projection(xyz) + self.time_projection(time.unsqueeze(-1)) + self.charge_projection(charge.unsqueeze(-1)) + self.aux_embd(aux) + self.pos_embd(pos)#+ self.sensor_embd(sensor_id)
        #x = self.sensor_embd(sensor_id) + self.time_projection(time.unsqueeze(-1))
        
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

class LatentCross(nn.Module):
    # https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/8cd1847a60b7937f26cebdf597d0131f0ea8dc7f/transformers4rec/torch/experimental.py
    def __init__(self, config):
        super().__init__()
        self.charge_projection = nn.Linear(1, config.hidden_size)
        self.aux_embd = nn.Embedding(3, config.hidden_size)
        
        self.layer_norm = LayerNorm(config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, seq_rep, charge, auxiliary):
        feature_rep = self.charge_projection(charge.unsqueeze(-1)) + self.aux_embd(auxiliary)
        feature_rep = self.layer_norm(feature_rep)
        feature_rep = self.dropout(feature_rep)
        
        out = torch.multiply(seq_rep, 1.0 + feature_rep)
        return out

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, hidden_states, mask):
        mask_expanded = mask.unsqueeze(-1).expand(hidden_states.shape)
        sum_hidden_states = torch.sum(hidden_states * mask_expanded, axis=1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_hidden_states / sum_mask
    
class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, hidden_states, mask):
        mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states[mask_expanded == 0.] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(hidden_states, 1)[0]
        return max_embeddings
    
class Pooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_pooling = MeanPooling()
        self.max_pooling = MaxPooling()
        
    def forward(self, hidden_states, mask):
        mean_pool = self.mean_pooling(hidden_states, mask)
        max_pool = self.max_pooling(hidden_states, mask)
        out = torch.cat((hidden_states[:, 0, :], mean_pool, max_pool), 1)
        return out

class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.input_layer = InputLayer(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, nhead=config.n_head, dropout=config.dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.n_layer)
        self.pooler = Pooler()
        #self.latent_cross = LatentCross(config)
        self.head = nn.Linear(3*config.hidden_size, 3)
        
        self.padding_value = config.padding_value
        self.ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
    def forward(self, inputs):
        padding_mask = inputs["auxiliary"] == self.padding_value
        x = self.input_layer(inputs["xyz"], inputs["time"], inputs["charge"], inputs["auxiliary"])#, inputs["sensor_id"])
        #x = self.input_layer(inputs["sensor_id"], inputs["time"])
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        #x = self.latent_cross(x, inputs["charge"], inputs["auxiliary"], inputs["xyz"])
        x = self.pooler(x, (~padding_mask).type(self.ptdtype))
        out = self.head(x)
        return out
    
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # https://github.com/karpathy/nanoGPT/blob/a82b33b525ca9855d705656387698e13eb8e8d4b/model.py
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.modules.activation.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
