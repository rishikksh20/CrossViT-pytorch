import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward, CrossAttention
import numpy as np



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self, small_dim = 96, small_depth = 4, small_heads =3, small_dim_head = 32, small_mlp_dim = 384,
                 large_dim = 192, large_depth = 1, large_heads = 3, large_dim_head = 64, large_mlp_dim = 768,
                 cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0.):
        super().__init__()
        self.transformer_enc_small = Transformer(small_dim, small_depth, small_heads, small_dim_head, small_mlp_dim)
        self.transformer_enc_large = Transformer(large_dim, large_depth, large_heads, large_dim_head, large_mlp_dim)

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, heads = cross_attn_heads, dim_head = large_dim_head, dropout = dropout)),
                nn.Linear(large_dim, small_dim),
                nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(small_dim, heads = cross_attn_heads, dim_head = small_dim_head, dropout = dropout)),
            ]))

    def forward(self, xs, xl):

        xs = self.transformer_enc_small(xs)
        xl = self.transformer_enc_large(xl)

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch

            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        return xs, xl





class CrossViT(nn.Module):
    def __init__(self, image_size, channels, num_classes, patch_size_small = 14, patch_size_large = 16, small_dim = 96,
                 large_dim = 192, small_depth = 1, large_depth = 4, cross_attn_depth = 1, multi_scale_enc_depth = 3,
                 heads = 3, pool = 'cls', dropout = 0., emb_dropout = 0., scale_dim = 4):
        super().__init__()

        assert image_size % patch_size_small == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_small = (image_size // patch_size_small) ** 2
        patch_dim_small = channels * patch_size_small ** 2

        assert image_size % patch_size_large == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_large = (image_size // patch_size_large) ** 2
        patch_dim_large = channels * patch_size_large ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        self.to_patch_embedding_small = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_small, p2 = patch_size_small),
            nn.Linear(patch_dim_small, small_dim),
        )

        self.to_patch_embedding_large = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size_large, p2=patch_size_large),
            nn.Linear(patch_dim_large, large_dim),
        )

        self.pos_embedding_small = nn.Parameter(torch.randn(1, num_patches_small + 1, small_dim))
        self.cls_token_small = nn.Parameter(torch.randn(1, 1, small_dim))
        self.dropout_small = nn.Dropout(emb_dropout)

        self.pos_embedding_large = nn.Parameter(torch.randn(1, num_patches_large + 1, large_dim))
        self.cls_token_large = nn.Parameter(torch.randn(1, 1, large_dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        self.multi_scale_transformers = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(MultiScaleTransformerEncoder(small_dim=small_dim, small_depth=small_depth,
                                                                              small_heads=heads, small_dim_head=small_dim//heads,
                                                                              small_mlp_dim=small_dim*scale_dim,
                                                                              large_dim=large_dim, large_depth=large_depth,
                                                                              large_heads=heads, large_dim_head=large_dim//heads,
                                                                              large_mlp_dim=large_dim*scale_dim,
                                                                              cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                                                                              dropout=dropout))

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head_small = nn.Sequential(
            nn.LayerNorm(small_dim),
            nn.Linear(small_dim, num_classes)
        )

        self.mlp_head_large = nn.Sequential(
            nn.LayerNorm(large_dim),
            nn.Linear(large_dim, num_classes)
        )


    def forward(self, img):

        xs = self.to_patch_embedding_small(img)
        b, n, _ = xs.shape

        cls_token_small = repeat(self.cls_token_small, '() n d -> b n d', b = b)
        xs = torch.cat((cls_token_small, xs), dim=1)
        xs += self.pos_embedding_small[:, :(n + 1)]
        xs = self.dropout_small(xs)

        xl = self.to_patch_embedding_large(img)
        b, n, _ = xl.shape

        cls_token_large = repeat(self.cls_token_large, '() n d -> b n d', b=b)
        xl = torch.cat((cls_token_large, xl), dim=1)
        xl += self.pos_embedding_large[:, :(n + 1)]
        xl = self.dropout_large(xl)

        for multi_scale_transformer in self.multi_scale_transformers:
            xs, xl = multi_scale_transformer(xs, xl)
        
        xs = xs.mean(dim = 1) if self.pool == 'mean' else xs[:, 0]
        xl = xl.mean(dim = 1) if self.pool == 'mean' else xl[:, 0]

        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)
        x = xs + xl
        return x
    
    
    

if __name__ == "__main__":
    
    img = torch.ones([1, 3, 224, 224])
    
    model = CrossViT(224, 3, 1000)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]

    
