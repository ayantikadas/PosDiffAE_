# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

#from pos_embed import get_2d_sincos_pos_embed
import numpy as np


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype='float')
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, apply_loss_on_unmasked: bool = False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE enco  der specifics
        self.in_chans = in_chans
        self.apply_loss_on_unmasked = apply_loss_on_unmasked
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) #qk_scale=None, 
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) #qk_scale=None,
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def grid_masking(self, x): #Rashmi ka observation, ismein mask ratio ka use nahi ho rha hai
        device = x.device
        N, L, D = x.shape  # batch, length, dim
        arr = np.random.random(x.shape[1])
        mask = []
        step = 4
        for index_ in range(0, len(arr), step):
            max_ = np.max(arr[index_: index_ + step])
            for index_2 in range(index_, index_ + step):
                mask.append(arr[index_2] != max_)  # 0 keep 1 remove

                
        mask = np.array(mask) * 1

        ids_keep = np.arange(x.shape[1])[mask == 0]

        ids_restore = np.concatenate((ids_keep, list(set(np.arange(L)) - set(ids_keep))))

        ids_keep = np.tile(ids_keep, (x.shape[0], 1))
        ids_keep = torch.tensor(ids_keep).to(device)

        ids_restore = np.tile(ids_restore, (x.shape[0], 1))
        ids_restore = torch.tensor(ids_restore).to(device)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len(ids_keep[0])] = 0

        # unshuffle to get the binary mask
        ids_restore_aux = torch.argsort(ids_restore, dim=1)
        mask = torch.gather(mask, dim=1, index=ids_restore_aux)

        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x, mask, ids_restore_aux
      
    def grid_masking_updated(self, x,index): #Rashmi ka observation, ismein mask ratio ka use nahi ho rha hai
        device = x.device
        N, L, D = x.shape  # batch, length, dim
        arr = np.random.random(x.shape[1])
        mask = []
#         step = 4
#         for index_ in range(0, len(arr), step):
#             max_ = np.max(arr[index_: index_ + step])
#             for index_2 in range(index_, index_ + step):
#                 mask.append(arr[index_2] != max_)  # 0 keep 1 remove
        i = 0
        count = 0
        while i<=(x.shape[1]-1): #This piece of logic is written by Rashmi
            if(i == index) and(count == 0):
                    mask.append(1)
                    mask.append(1)
                    mask.append(1)
                    mask.append(1)
                    i = i+4
                    index = index+14
                    count = 1
            elif(i == index)and(count == 1):
                    mask.append(1)
                    mask.append(1)
                    mask.append(1)
                    mask.append(1)
                    i = i+4
                    index = index+14
                    count = 2
            elif(i == index)and(count == 2):
                    mask.append(1)
                    mask.append(1)
                    mask.append(1)
                    mask.append(1)
                    i = i+4
                    index = index+14
                    count = 3
            elif(i == index)and(count == 3):
                    mask.append(1)
                    mask.append(1)
                    mask.append(1)
                    mask.append(1)
                    index = index+14
                    i = i+4
                    count = 4
            else:
                mask.append(0)
                i = i+1
                
        mask = np.array(mask) * 1
        
        #print(mask)
        #print(mask.shape)
        
        ids_keep = np.arange(x.shape[1])[mask == 0]

        ids_restore = np.concatenate((ids_keep, list(set(np.arange(L)) - set(ids_keep))))

        ids_keep = np.tile(ids_keep, (x.shape[0], 1))
        ids_keep = torch.tensor(ids_keep).to(device)

        ids_restore = np.tile(ids_restore, (x.shape[0], 1))
        ids_restore = torch.tensor(ids_restore).to(device)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len(ids_keep[0])] = 0

        # unshuffle to get the binary mask
        ids_restore_aux = torch.argsort(ids_restore, dim=1)
        mask = torch.gather(mask, dim=1, index=ids_restore_aux)

        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x, mask, ids_restore_aux
    def forward_encoder(self, x, mask_ratio, is_testing=False, idx_masking=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio

#         if is_testing:
#             x, mask, ids_restore = self.grid_masking(x)
#             # import pdb; pdb.set_trace()
#         else:
            #x, mask, ids_restore = self.grid_masking_updated(x,116)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
#             plt.imshow(x[0].detach().numpy())
            #print(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        #print("decoder output understanding:")
        #print(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
            #plt.imshow(x[0].detach().numpy())
            #print(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
#         print(imgs[0].shape)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
#Need to re-think how to utilise ssim loss        
#         y1 = model.unpatchify(pred)
#         #y1 = torch.einsum('nchw->nhwc', y).detach().cpu()
#         print(y1.shape)
#         y1 = y1.detach().numpy()
#         ssim_loss = ssim(y1[0][0], imgs[0][0], data_range = imgs[0].max()-imgs[0].min())
#         print(ssim_loss)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        if self.apply_loss_on_unmasked:
            loss = loss.mean() 
        else:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    def forward(self, imgs, mask_ratio=0.75, is_testing=False, idx_masking=None, return_latent: bool = True):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, is_testing, idx_masking)
#         pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
#         loss = self.forward_loss(imgs, pred, mask)
#         if return_latent:
#             return loss, pred, mask, latent[:, 1:, :]
#         else:
        return latent

def model_call(image_batch=torch.ones((1, 3, 256, 256)),model_weight = ' '):    
    device = 'cuda'
    model_mae =  MaskedAutoencoderViT()
    model_mae.load_state_dict(torch.load(model_weight))
    model_mae.eval()
    model_mae.to(device);
    cond = model_mae(image_batch.to(device))
    return model_mae, cond

    return model_mae, cond
if __name__ == '__main__':
    model_mae = MaskedAutoencoderViT()
#     model_mae.load_state_dict(torch.load('/storage/Ayantika/Masked_auto_encoder/MAE-code/ckpt_official/epoch_400.pt'))


#     def forward(self, imgs, mask_ratio=0.75, is_testing=False, idx_masking=None, return_latent: bool = True):
#         latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, is_testing, idx_masking)
#         pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
#         loss = self.forward_loss(imgs, pred, mask)
#         if return_latent:
#             return loss, pred, mask, latent[:, 1:, :]
#         else:
#             return loss, pred, mask

# def model_call(image_batch=torch.ones((1, 3, 256, 256))):    
#     device = 'cuda'
#     model_mae =  MaskedAutoencoderViT()
#     model_mae.load_state_dict(torch.load('/storage/Ayantika/Masked_auto_encoder/MAE-code/ckpt_official/epoch_400.pt'))
#     model_mae.eval()
#     model_mae.to(device);
#     _,_,_,cond = model_mae(image_batch.to(device))
#     return model_mae, cond

#     return model_mae, cond
# if __name__ == '__main__':
#     model_mae = MaskedAutoencoderViT()
#     model_mae.load_state_dict(torch.load('/storage/Ayantika/Masked_auto_encoder/MAE-code/ckpt_official/epoch_400.pt'))

