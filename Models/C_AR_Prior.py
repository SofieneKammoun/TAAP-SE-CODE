import torch
from torch import nn 
from einops import    repeat 
from Models.Conformer import Conformer



def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class C_AR_Model_Prior(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        dim,
        max_seq_len,
        N_layers,
        k=8,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0,
        conv_kernel_size=10
    ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        self.dim = dim
        self.input_dim=input_dim
        self.spatial_pos_emb = nn.Embedding(int(max_seq_len )+ 1,self.dim) # account for a boundary case
        self.Conformer_Prior = Conformer(
            dim = self.dim,
            layers = N_layers,
            dim_head = dim_head,
            heads = heads,
            conv_kernel_size = conv_kernel_size,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult,
            conv_causal=True
        )
        self.spatial_start_token = nn.Parameter(torch.randn(self.dim))

        self.input_layer  = nn.Linear(self.input_dim, self.dim)

        self.output_layer = nn.Linear(self.dim, self.input_dim)
        self.Var_layer = nn.Linear(self.dim, self.input_dim)
        self.W_raw = nn.Parameter(torch.eye(input_dim))
        
        


    def forward(self,  clean_embeds,return_loss = False):
        
        assert clean_embeds.ndim == 3
        b, T, _, device = *clean_embeds.shape, clean_embeds.device
        assert T <= (self.max_seq_len ), f'spatial dimension T = {T} is greater than the max_seq_len {self.max_seq_len} set'
        
        if T==0 :
            clean_tokens=torch.empty((b,0,self.dim), device=device)
        else:
            clean_tokens = self.input_layer(clean_embeds)

        spatial_pos = self.spatial_pos_emb(torch.arange(T+1, device = device))

         # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions
        
        spatial_tokens = torch.cat((
            repeat(self.spatial_start_token, 'f -> b 1 f', b = b),
            clean_tokens
        ), dim = -2)
        spatial_tokens = spatial_tokens + spatial_pos 
        spatial_tokens = self.Conformer_Prior(spatial_tokens)
        
        

# #### ============= FULL Matrix GAUSSIAN MODEL =============

        mean = self.output_layer(spatial_tokens)
        log_var= self.Var_layer(spatial_tokens)
        
        if not return_loss :
            return mean,log_var
        W, _ = torch.linalg.qr(self.W_raw)
        mean = mean[:,:-1,:]
        log_var =log_var[:,:-1,:]
        labels = clean_embeds
        
        
        diff = labels - mean     # (B,T,D)
        
        
        diff_tilde = torch.einsum(
            'ij,btj->bti', W.T, diff
                )   
        var = torch.exp(log_var)
        
        nll = 0.5 * (
            (diff_tilde**2 / var).sum(dim=-1)
            + log_var.sum(dim=-1)
        )
        
        loss = nll.mean()
        
            
        return loss
