import sys
sys.path.insert(0,'/storage/Ayantika/Transunet/TransUNet/')
import random
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
import torch
import numpy as np
import os

class vit_model_call():
    def __init__(self, device = 'cuda', model_weight='/storage/Ayantika/Transunet/TransUNet/Weights/ckpt/segment_0.0012858885072417699.pt',\
                 ):
        num_classes=3
        max_iterations=1200
        max_epochs=20
        batch_size=24
        n_gpu='deterministic'
        base_lr = 0.01
        img_size = 256
        seed = 1234
        n_skip = 3
        vit_name='R50-ViT-B_16'
        vit_patches_size=16
        if not n_gpu=='deterministic':
            cudnn.benchmark = True
            cudnn.deterministic = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        dataset_name = 'Synapse'
        img_size=256
        is_pretrain = True
        exp = 'TU_' + dataset_name + str(img_size)
        snapshot_path = "../model/{}/{}".format(exp, 'TU')
        snapshot_path = snapshot_path + '_pretrain' if is_pretrain else snapshot_path
        snapshot_path += '_' + vit_name
        snapshot_path = snapshot_path + '_skip' + str(n_skip)
        snapshot_path = snapshot_path + '_vitpatch' + str(vit_patches_size) if vit_patches_size!=16 else snapshot_path
        snapshot_path = snapshot_path+'_'+str(max_iterations)[0:2]+'k' if max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_epo' +str(max_epochs) if max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path+'_bs'+str(batch_size)
        snapshot_path = snapshot_path + '_lr' + str(base_lr) if base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_'+str(img_size)
        snapshot_path = snapshot_path + '_s'+str(seed) if seed!=1234 else snapshot_path
        self.device = device
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = num_classes
        config_vit.n_skip = n_skip
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        self.model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
        self.model_weight = model_weight
#         self.image_batch = image_batch
        
    def model_call(self,image_batch=torch.ones((1, 3, 256, 256))):      

        #model.load_from(weights=np.load('/data/Transunet/Transunet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
#         self.model= 
        self.model.load_state_dict(torch.load(self.model_weight))
#         if self.device=='cuda':
        self.model.eval()
        self.model = self.model.to(self.device)
        output = self.model.transformer(image_batch.to(self.device))
        
        return self.model,output[0]
        