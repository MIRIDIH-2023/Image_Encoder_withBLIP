import torch
import torch.nn as nn
from aggregation_block import AggregationBlock

class SetPredictionModule(nn.Module):
    def __init__(
        self, 
        num_embeds, 
        d_in, 
        d_out, 
        axis, 
        pos_enc, 
        query_dim = 512 ,
    ):
        super(SetPredictionModule, self).__init__()
        self.num_embeds = num_embeds
        self.residual_norm = nn.LayerNorm(d_out) if True else nn.Identity()
        self.res_act_local_dropout = nn.Dropout(0.1)
        self.res_act_global_dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(d_out, 1024) if True else nn.Identity()
        
        self.agg_block = AggregationBlock(
            depth = 4,
            input_channels = d_in,
            input_axis = axis,
            num_latents = num_embeds,
            latent_dim = query_dim,
            num_classes = d_out,
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            weight_tie_layers = False,
            pos_enc_type = pos_enc,
            pre_norm = True,
            post_norm= False,
            activation = 'gelu',
            last_ln = True,
            ff_mult = 4,
            more_dropout = False,
            xavier_init = True,
            query_fixed = False,
            query_xavier_init = True,
            query_type = 'slot',
            first_order=True
        )
        
    def forward(self, local_feat, global_feat=None, pad_mask=None, lengths=None):
        set_prediction = self.agg_block(local_feat, mask=pad_mask)
        set_prediction = self.res_act_local_dropout(set_prediction)
        global_feat = global_feat.unsqueeze(1).repeat(1, self.num_embeds, 1)
        out = self.residual_norm(self.res_act_global_dropout(global_feat)) + set_prediction
        out = self.fc(out)
        
        return out, None, set_prediction

test_model = SetPredictionModule(512,1,1,1,'sine',1)

test_input = torch.randn((1,1,512))
test_model(test_input)