
import sys
sys.path.append('/content/drive/MyDrive')
sys.path.append('/mnt/c/GridEyeS')
import argparse
import os
import time
from itertools import product

import numpy as np

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from MambaCD.changedetection.datasets.make_data_loader import  make_data_loader, ONERAChangeDetectionDataset
from MambaCD.changedetection.utils_func.metrics import Evaluator

from MambaCD.changedetection.models.MambaBCD_multimodal_diff import STMambaBCD_multimodal 
from MambaCD.changedetection.models.MambaBCD import STMambaBCD
from MambaCD.changedetection.comparison_models.HANet import HAN
from MambaCD.changedetection.comparison_models.Unet import Unet
from MambaCD.changedetection.comparison_models.SiamUnet_diff import SiamUnet_diff
from MambaCD.changedetection.comparison_models.siamunet_conc import SiamUnet_conc_multi
from MambaCD.changedetection.comparison_models.ChangeFormer import ChangeFormerV4, ChangeFormerV5, ChangeFormerV6
from MambaCD.changedetection.comparison_models.base_transformer import BASE_Transformer
from MambaCD.changedetection.comparison_models.CGNet import CGNet
#from MambaCD.changedetection.comparison_models.CDMamba import CDMamba
import MambaCD.changedetection.utils_func.lovasz_loss as L



# Add TensorBoard import
from torch.utils.tensorboard import SummaryWriter



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

            
class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)
        
        # Define channel counts
        self.sar_channels = 2  # SAR (s1) has 2 channels: HH and VV
        self.opt_channels = args.opt_bands  # Number of optical bands
        
        # Determine input channels based on modality
        if args.sar_only:
            # SAR (s1) has 2 channels (HH, VV) for both pre and post
            self.in_channels = self.sar_channels  # (HH,VV) × (pre,post) = 4 channels
            print(f"Using SAR-only mode (HH,VV channels) with {self.in_channels} total input channels")
        elif args.opt_only:
            # Optical (s2) bands for pre and post
            self.in_channels = self.opt_channels 
            print(f"Using Optical-only mode with {self.in_channels} total input channels")
        else:
            # Both SAR (HH,VV) and Optical bands for pre and post
            self.in_channels = (self.sar_channels + self.opt_channels)
            print(f"Using both SAR (HH,VV) and Optical with {self.in_channels} total input channels")
            

        self.evaluator = Evaluator(num_class=2)
        models = {
            'STMambaBCD': STMambaBCD,
            'STMambaBCD_multimodal': STMambaBCD_multimodal,
            'HAN': HAN,
            'Unet': Unet,
            'SiamUnet_diff': SiamUnet_diff,
            'SiamUnet_conc_multi': SiamUnet_conc_multi,
            'ChangeFormerV4': ChangeFormerV4,
            'ChangeFormerV5': ChangeFormerV5,
            'ChangeFormerV6': ChangeFormerV6,
            'BASE_Transformer': BASE_Transformer,
            'CGNet': CGNet,
            #'CDMamba': CDMamba
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
            if args.model in ['Unet', 'SiamUnet_diff', 'ChangeFormerV4','ChangeFormerV5','ChangeFormerV6', 'HAN']:
                self.deep_model = model_class(self.in_channels, 2)
           # elif args.model == 'CDMamba':
            #    self.deep_model = model_class(in_channels = 15, out_channels =2)
            elif args.model == 'CGNet':
                self.deep_model = model_class()
            elif args.model == 'SiamUnet_conc_multi':
                self.deep_model = model_class((self.opt_channels, 2), 2)
            elif args.model == 'BASE_Transformer':
                self.deep_model = model_class(input_nc=self.in_channels, output_nc=2, token_len=4, 
                                           resnet_stages_num=4, with_pos='learned', 
                                           enc_depth=1, dec_depth=8, decoder_dim_head=8)

        self.deep_model = self.deep_model.cuda()
        num_params = count_parameters(self.deep_model)
        print(f"The model has {num_params:,} trainable parameters")
        
        # Create model save path with modality information
        modality = 'SAR_HHVV' if args.sar_only else ('OPT' if args.opt_only else 'FULL')
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                          f"{args.model}_{modality}")
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.writer = SummaryWriter(log_dir=os.path.join(self.model_save_path, 'tensorboard_logs'))

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

    def training(self):
        self.train_data_loader, self.weights = make_data_loader(self.args, shuffle=True)
        self.val_data_loader, _ = make_data_loader(self.args, split='val', shuffle=False)
        
        self.optim = optim.AdamW(self.deep_model.parameters(),
                                lr=self.args.learning_rate,
                                weight_decay=self.args.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.95)

        best_kc = 0.0
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)

        for epoch in range(self.args.epochs):
            self.deep_model.train()
            train_loss = 0.0
            self.evaluator.reset()
            
            for i, data in enumerate(tqdm(self.train_data_loader)):
                # Data loader returns appropriate patches based on modality setting
                if self.args.sar_only or self.args.opt_only:
                    pre_patch, post_patch, labels, _, _, _ = data
                    pre_change_imgs = pre_patch
                    post_change_imgs = post_patch
                else:
                    s1_pre_patch, s1_post_patch, s2_pre_patch, s2_post_patch, labels, _, _, _ = data
                    pre_change_imgs = torch.cat((s2_pre_patch, s1_pre_patch), dim=1)
                    post_change_imgs = torch.cat((s2_post_patch, s1_post_patch), dim=1)
                
                labels = labels.cuda().long()
                
                self.optim.zero_grad()

                if self.args.model == 'SiamUnet_conc_multi':
                        output = self.deep_model(s2_pre_patch.cuda().float(), s2_post_patch.cuda().float(),s1_pre_patch.cuda().float(), s1_post_patch.cuda().float())
                else: 
                    pre_change_imgs = pre_change_imgs.cuda().float()
                    post_change_imgs = post_change_imgs.cuda().float()
                    output = self.deep_model(pre_change_imgs, post_change_imgs)
               
                ce_loss = F.cross_entropy(output, labels, ignore_index=255, weight=self.weights.cuda())
                lovasz_loss = L.lovasz_softmax(F.softmax(output, dim=1), labels, ignore=255)
                main_loss = ce_loss + self.args.lovasz_weight * lovasz_loss
                
                main_loss.backward()
                train_loss += main_loss.item()
                self.optim.step()

                _, predicted = torch.max(output.data, 1)
                predicted = predicted.int().cpu().numpy()
                labels = labels.int().cpu().numpy()
                self.evaluator.add_batch(labels, predicted)

            avg_train_loss = train_loss / len(self.train_data_loader)
            train_metrics = self.get_metrics()
            print(f'Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}')
            print(train_metrics)
            
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            for metric_name, metric_value in train_metrics.items():
                self.writer.add_scalar(f'Metrics/train/{metric_name}', metric_value, epoch)
            
            val_loss, val_metrics = self.validation()
            print(f'Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}')
            print(val_metrics)
            
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Metrics/validation/{metric_name}', metric_value, epoch)
            
            self.scheduler.step()
            self.writer.add_scalar('Learning_rate', self.scheduler.get_last_lr()[0], epoch)
            
            current_kc = val_metrics['F1 score']
            if current_kc > best_kc:
                best_kc = current_kc
                modality = 'SAR_HHVV' if self.args.sar_only else ('OPT' if self.args.opt_only else 'FULL')
                model_name = f'best_model_{modality}.pth'
                torch.save(self.deep_model.state_dict(),
                          os.path.join(self.model_save_path, model_name))
                print(f"Saved best model: {model_name}")

            early_stopping(val_metrics['F1 score'])
            if early_stopping.early_stop:
                print("Early stopping triggered")
                self.writer.close()
                break

        return best_kc

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        
        total_val_loss = 0.0
        num_batches = 0
        
        self.deep_model.eval()
        with torch.no_grad():
            for data in tqdm(self.val_data_loader):
                # Data loader returns appropriate patches based on modality setting
                if self.args.sar_only or self.args.opt_only:
                    pre_patch, post_patch, labels, _, _, _ = data
                    pre_change_imgs = pre_patch
                    post_change_imgs = post_patch
                else:
                    s1_pre_patch, s1_post_patch, s2_pre_patch, s2_post_patch, labels, _, _, _ = data
                    pre_change_imgs = torch.cat((s2_pre_patch, s1_pre_patch), dim=1)
                    post_change_imgs = torch.cat((s2_post_patch, s1_post_patch), dim=1)

                labels = labels.cuda().long()

                if self.args.model == 'SiamUnet_conc_multi':
                        output = self.deep_model(s2_pre_patch.cuda().float(), s2_post_patch.cuda().float(),s1_pre_patch.cuda().float(), s1_post_patch.cuda().float())
                else: 

                    pre_change_imgs = pre_change_imgs.cuda().float()
                    post_change_imgs = post_change_imgs.cuda().float()
                    output = self.deep_model(pre_change_imgs, post_change_imgs)
                
                ce_loss = F.cross_entropy(output, labels, ignore_index=255) 
                lovasz_loss = L.lovasz_softmax(F.softmax(output, dim=1), labels, ignore=255)
                val_loss = ce_loss + self.args.lovasz_weight * lovasz_loss
                
                total_val_loss += val_loss.item()
                num_batches += 1

                _, predicted = torch.max(output.data, 1)
                predicted = predicted.int().cpu().numpy()
                labels = labels.int().cpu().numpy()
                self.evaluator.add_batch(labels, predicted)
        
        avg_val_loss = total_val_loss / num_batches
        val_metrics = self.get_metrics()
        
        return avg_val_loss, val_metrics

    def get_metrics(self):
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy_Class()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        
        metrics = {
            f"Recall rate": rec,
            f"Precision rate": pre,
            f"BA": oa,
            f"F1 score": f1_score,
            f"IoU": iou,
            f"Kappa coefficient": kc
        }
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Training on ONERA Multimodal Dataset")
    parser.add_argument('--cfg', type=str, default='/mnt/c/GridEyeS/MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='ONERA')
    parser.add_argument('--dataset_path', type=str, default=r'C:\GridEyeS\Data\ONERA')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--model_param_path', type=str, default='../saved_models')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='STMambaBCD')
    parser.add_argument('--lovasz_weight', type=float, default=0.7)
    parser.add_argument('--sar_only', type=bool, default=False)
    parser.add_argument('--opt_only', type=bool, default=False)
    parser.add_argument('--opt_bands', type=int, default=13)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.training()

if __name__ == "__main__":
    main()