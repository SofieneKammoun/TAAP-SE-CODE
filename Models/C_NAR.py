import torch
from torch import nn 
from einops import  repeat
from Models.Conformer import Conformer

def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class C_NAR_Model(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        dim,
        max_seq_len,
        N_layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0,
        conv_kernel_size=10,
        causal=False
    ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        self.dim = dim
        self.input_dim=input_dim
        self.spatial_pos_emb = nn.Embedding(max_seq_len + 1,self.dim) # Not used here
        self.causal=causal
        self.noise_transformer = Conformer(
            dim = self.dim,
            layers = N_layers,
            dim_head = dim_head,
            heads = heads,
            conv_kernel_size = conv_kernel_size,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult,
            conv_causal=self.causal
        )

        self.input_layer  = nn.Linear(self.input_dim, self.dim)
        self.output_layer = nn.Linear(self.dim, self.input_dim)
    def forward_empty(self, batch_size,noisy):
        spatial_tokens = repeat(self.spatial_start_token, 'd -> b 1 d', b = batch_size)
        logits = self.spatial_transformer(spatial_tokens)

        return logits

    def forward(self, embeds, clean_embeds=None , return_loss = False,gen=False):
        assert embeds.ndim == 3
        if embeds.numel() == 0:
            return self.forward_empty(embeds.shape[0])
        tokens = self.input_layer(embeds)
        spatial_tokens = tokens  
        
        
        spatial_tokens = self.noise_transformer(spatial_tokens)
        out_embeds = self.output_layer(spatial_tokens)
        
        if not return_loss :
           
            return out_embeds 
        
        preds = out_embeds 
        labels = clean_embeds
        loss = torch.nn.functional.mse_loss(preds, labels)
        return loss


