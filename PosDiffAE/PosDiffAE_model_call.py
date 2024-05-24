import sys

import pandas as pd
import torch
from torch.utils.data import Dataset
from templates import *
# from templates_laten import *
from omegaconf import OmegaConf
from choices import *
from experiment_ import LitModel

from torch.utils.data import DataLoader
from config_ import TrainConfig
class PosDiffAE_model_call():
    def __init__(self, device = 'cuda', model_weight=' ',\
                 ):
        gpus = [0]
        conf = ffhq128_autoenc_130M()
        data_config_path = os.path.abspath('././PosDiffAE_model_call.py').split('PosDiffAE_model_call')[0]+'PosDiffAE/'+'/config_file_.yaml'
        #config = OmegaConf.load('/storage/Ayantika/Diffusion_AE_hist_pathology/config_file_ADNI.yaml')
        #     data_config_path = '/storage/Ayantika/Diffusion_AE_hist_pathology/config_file_ADNI.yaml'
        #     config = OmegaConf.load('/storage/Ayantika/Diffusion_AE_hist_pathology/config_file_ADNI.yaml')

        conf.batch_size = 1
        conf.data_name =  'hist_path'
        #     every_n_train_steps = conf.save_every_samples
        # train(conf, gpus=gpus)
        conf.img_size = 256
        conf.model_conf.image_size = 256
        #     print("conf.model_conf.image_size",conf.model_conf.image_size)
        conf.model_conf.in_channels = 3
        conf.model_conf.out_channels = 3 
        #     conf.base_dir = '/storage/Ayantika/Diffusion_AE_hist_pathology/checkpoints_hist_path_with_r_theta/'
        if not os.path.exists(conf.base_dir):
            os.mkdir(conf.base_dir)
        conf.name = 'Nissl' 
        conf.sample_size = 10 ### should be less than batch size
        conf.batch_size_eval = 10

        conf.beatgans_loss_type = LossType.mse
        conf.beatgans_model_mean_type = ModelMeanType.start_x

        conf.data_config_path = data_config_path
        conf.img_size_height = 256
        conf.img_size_width = 256
        conf.eval_num_images = 800
        self.conf = conf
        self.model_weight = model_weight
        self.device = device
        
    def model_call(self,image_batch=torch.ones((1, 3, 256, 256))):    
        

        model = LitModel(self.conf)
        
        state = torch.load(self.model_weight,\
                       map_location=self.device)
        model.load_state_dict(state['state_dict'], strict=False)
        model.ema_model.eval()
        model.ema_model.to(self.device);
        cond = model.encode(image_batch.to(self.device))
        
        return model, cond