import sys
sys.path.append('/content/drive/MyDrive')
sys.path.append('/mnt/c/GridEyeS')

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import imageio
from collections import defaultdict

from MambaCD.changedetection.configs.config import get_config
from MambaCD.changedetection.datasets.make_data_loader import  make_data_loader, ONERAChangeDetectionDataset
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.MambaBCD import STMambaBCD
from MambaCD.changedetection.models.MambaBCD_multimodal_diff import STMambaBCD_multimodal
import MambaCD.changedetection.utils_func.lovasz_loss as L
from MambaCD.changedetection.models.MambaBCD import STMambaBCD

from MambaCD.changedetection.comparison_models.HANet import HAN
from MambaCD.changedetection.comparison_models.Unet import Unet

from MambaCD.changedetection.comparison_models.SiamUnet_diff import SiamUnet_diff
from MambaCD.changedetection.comparison_models.siamunet_conc import SiamUnet_conc_multi
from MambaCD.changedetection.comparison_models.ChangeFormer import ChangeFormerV6
from MambaCD.changedetection.comparison_models.base_transformer import BASE_Transformer

class Inference(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.evaluator = Evaluator(num_class=2)

        # Define channel counts
        self.sar_channels = 2  # SAR (s1) has 2 channels: HH and VV
        self.opt_channels = args.opt_bands  # Number of optical bands
        
        # Determine input channels based on modality
        if args.sar_only:
            # SAR (s1) has 2 channels (HH, VV) for both pre and post
            self.in_channels = self.sar_channels * 2  # (HH,VV) × (pre,post) = 4 channels
            print(f"Using SAR-only mode (HH,VV channels) with {self.in_channels} total input channels")
        elif args.opt_only:
            # Optical (s2) bands for pre and post
            self.in_channels = self.opt_channels * 2
            print(f"Using Optical-only mode with {self.in_channels} total input channels")
        else:
            # Both SAR (HH,VV) and Optical bands for pre and post
            self.in_channels = (self.sar_channels + self.opt_channels) * 2
            print(f"Using both SAR (HH,VV) and Optical with {self.in_channels} total input channels")
            print(f"- SAR channels: {self.sar_channels * 2} (HH,VV × pre,post)")
            print(f"- Optical channels: {self.opt_channels * 2} (bands × pre,post)")
            
       # Dictionary mapping model names to their classes
        models = {
            'STMambaBCD': STMambaBCD,
            'STMambaBCD_multimodal': STMambaBCD_multimodal,
            'HAN': HAN,
            'Unet': Unet,
            'SiamUnet_diff': SiamUnet_diff,
            'SiamUnet_conc_multi': SiamUnet_conc_multi,
            'ChangeFormerV6': ChangeFormerV6,
            'BASE_Transformer': BASE_Transformer
        }
        
        if args.model not in models:
            raise ValueError(f"Model {args.model} not found. Available models: {list(models.keys())}")
            
        model_class = models[args.model]
        
        # Initialize model with correct input channels
        if args.model in ['STMambaBCD', 'STMambaBCD_multimodal']:
            self.deep_model = model_class(
                pretrained=args.pretrained_weight_path,
                patch_size=config.MODEL.VSSM.PATCH_SIZE,
                #in_chans=self.in_channels,  # Total input channels based on modality
                
                num_classes=config.MODEL.NUM_CLASSES,
                depths=config.MODEL.VSSM.DEPTHS,
                dims=config.MODEL.VSSM.EMBED_DIM,
                ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
                ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
                ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
                ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
                ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
                ssm_conv=config.MODEL.VSSM.SSM_CONV,
                ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
                ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
                ssm_init=config.MODEL.VSSM.SSM_INIT,
                forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
                mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
                mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
                mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                patch_norm=config.MODEL.VSSM.PATCH_NORM,
                norm_layer=config.MODEL.VSSM.NORM_LAYER,
                downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
                patchembed_version=config.MODEL.VSSM.PATCHEMBED,
                gmlp=config.MODEL.VSSM.GMLP,
                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                opt_only = args.opt_only,
                sar_only = args.sar_only,
                opt_bands = args.opt_bands
            )
        else:
            # Initialize other models with correct input channels
            if args.model in ['Unet', 'SiamUnet_diff', 'ChangeFormerV6', 'HAN']:
                self.deep_model = model_class(self.in_channels//2, 2)
            elif args.model == 'SiamUnet_conc_multi':
                self.deep_model = model_class((self.opt_channels, 2), 2)
            elif args.model == 'BASE_Transformer':
                self.deep_model = model_class(input_nc=self.in_channels, output_nc=2, token_len=4, 
                                           resnet_stages_num=4, with_pos='learned', 
                                           enc_depth=1, dec_depth=8, decoder_dim_head=8)

        self.deep_model = self.deep_model.cuda()
        self.epoch = args.max_iters // args.batch_size

        self.change_map_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model, 'change_map')

        if not os.path.exists(self.change_map_saved_path):
            os.makedirs(self.change_map_saved_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.deep_model.eval()

    def infer(self):
        torch.cuda.empty_cache()
        dataset = ONERAChangeDetectionDataset(self.args.dataset_path, 'val', patch_size=self.args.patch_size, stride=int(self.args.patch_size/8)-1,
                                              sar_only=self.args.sar_only, opt_only=self.args.opt_only, opt_bands=self.args.opt_bands)
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        self.evaluator.reset()

        predictions_by_city = defaultdict(list)
        labels_by_city = defaultdict(list)

        with torch.no_grad():
            for itera, data in enumerate(tqdm(val_data_loader, desc="Processing patches")):
                if self.args.sar_only or self.args.opt_only:
                    pre_patch, post_patch, labels, names, _, _ = data
                    pre_change_imgs = pre_patch
                    post_change_imgs = post_patch
                else:
                    s1_pre_patch, s1_post_patch, s2_pre_patch, s2_post_patch, labels, names, _, _ = data

                   
                    pre_change_imgs = torch.cat((s2_pre_patch, s1_pre_patch), dim=1)
                    post_change_imgs = torch.cat((s2_post_patch, s1_post_patch), dim=1)
                
                labels = labels.cuda().long()

                if self.args.model == 'SiamUnet_conc_multi':
                        output = self.deep_model(s2_pre_patch.cuda().float(), s2_post_patch.cuda().float(),s1_pre_patch.cuda().float(), s1_post_patch.cuda().float())
                else: 

                    pre_change_imgs = pre_change_imgs.cuda().float()
                    post_change_imgs = post_change_imgs.cuda().float()
                    output = self.deep_model(pre_change_imgs, post_change_imgs)
                _, predicted = torch.max(output.data, 1)
                predicted = predicted.int().cpu().numpy().squeeze()  
                labels = labels.int().cpu().numpy().squeeze()  # Remove channel dimension
                self.evaluator.add_batch(labels, predicted)
                city = names[0].split('_')[0]  # Assuming the city name is the first part of the image name
                predictions_by_city[city].append(predicted)

                labels_by_city[city].append(labels)

                
        print("Stitching patches and evaluating...")
        for city in tqdm(dataset.cities, desc="Processing cities"):
            stitched_prediction = dataset.stitch_patches(predictions_by_city[city], city).astype(np.int64)
            stitched_label = dataset.processed_data[city]['label'].astype(np.int64)
            #self.evaluator.add_batch(stitched_label, stitched_prediction)

            binary_change_map = (stitched_prediction > 0.5).astype(np.uint8) * 255
            imageio.imwrite(os.path.join(self.change_map_saved_path, f'{city}_change_map.png'), binary_change_map)

            # save plot of difference maps
            DI = np.stack((255*stitched_label,255*stitched_prediction,255*stitched_label),2)
            imageio.imwrite(os.path.join(self.change_map_saved_path, f'{city}_diff_cm.png'), DI.astype(np.uint8))

        f1_score = self.evaluator.Pixel_F1_score()
        ba = self.evaluator.Pixel_Accuracy_Class()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Recall rate is {rec}, Precision rate is {pre}, Class Accuracy is {ba}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')         
        print('Inference stage is done!')

def main():
    parser = argparse.ArgumentParser(description="Inference on ONERA dataset")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='LEVIR-CD+')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=224)
    #parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaBCD')
    parser.add_argument('--result_saved_path', type=str, default='MambaBCD/results')
    parser.add_argument('--model', type=str, default='STMambaBCD')
    parser.add_argument('--sar_only', type=bool, default=False)
    parser.add_argument('--opt_only', type=bool, default=False)
    parser.add_argument('--opt_bands', type=int, default=13)


    parser.add_argument('--resume', type=str)

    args = parser.parse_args()

    infer = Inference(args)
    infer.infer()

if __name__ == "__main__":
    main()