import sys
sys.path.append('/content/drive/MyDrive')
sys.path.append(r"C:\GridEyeS")
import argparse
import os

import imageio
import numpy as np
#from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
from collections import defaultdict
import torch
import matplotlib.pyplot as plt

import MambaCD.changedetection.datasets.imutils as imutils


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img



class ONERAChangeDetectionDataset(Dataset):
    def __init__(self, dataset_path, type='train', patch_size=224, stride=56, opt_only=False, sar_only=False, opt_bands=13):
        if opt_only and sar_only:
            raise ValueError("Cannot set both opt_only and sar_only to True")
        self.dataset_path = dataset_path
        self.type = type
        self.patch_size = patch_size
        self.stride = stride
        self.patches_info = self.preprocess_dataset()
        self.cities = self.extract_cities()
        self.opt_only = opt_only
        self.sar_only = sar_only
        self.opt_bands = opt_bands
        self.processed_data = self.process_all_images()
        self.fp_modifier = 10
        self.weights = self.calculate_class_weights()
        
        

    


    def extract_cities(self):
        return list(set(patch['city'] for patch in self.patches_info))

    def preprocess_dataset(self):
        cache_file = f'{self.type}_patches_info.json'
        patches_info = []
        label_folder = f'{self.type.capitalize()} Labels'

        for city in os.listdir(os.path.join(self.dataset_path, label_folder)):
            city_path = os.path.join(self.dataset_path, label_folder, city)
            if os.path.isdir(city_path):
                patches_info.extend(self.create_patches(city))

        with open(cache_file, 'w') as f:
            json.dump(patches_info, f)

        return patches_info

    def create_patches(self, city):
        patches = []
        label_folder = f'{self.type.capitalize()} Labels'
        label_path = os.path.join(self.dataset_path, label_folder, city, 'cm')
        label_file = next(f for f in os.listdir(label_path) if f.endswith('-cm.tif'))
        label = imageio.imread(os.path.join(label_path, label_file))

        h, w = label.shape[:2]
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patches.append({'city': city, 'top': i, 'left': j})
        return patches

    def load_image(self, folder_path):
        image_file = os.listdir(folder_path)[0]
        return imageio.imread(os.path.join(folder_path, image_file))

    def load_and_stack_tiff(self, folder_path, select_list=None):
        tiff_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])
        if select_list:
            return np.stack([imageio.imread(os.path.join(folder_path, f+'.tif')) for f in select_list], axis=-1)
        else:
            return np.stack([imageio.imread(os.path.join(folder_path, f)) for f in tiff_files], axis=-1)

    def process_MS(self, img):
        intensity_min, intensity_max = 0, 10000
        img = np.clip(img, intensity_min, intensity_max)
        return self.rescale(img, intensity_min, intensity_max)

    def process_SAR(self, img):
        dB_min, dB_max = -25, 0
        img = np.clip(img, dB_min, dB_max)
        return self.rescale(img, dB_min, dB_max)

    def rescale(self, img, oldMin, oldMax):
        oldRange = oldMax - oldMin
        return np.float32((img - oldMin) / oldRange)

    def transforms(self, aug, s1_pre_patch=None, s1_post_patch=None, s2_pre_patch=None, s2_post_patch=None, label=None):
        if aug:
            if self.sar_only:
                s1_channels = s1_pre_patch.shape[-1]
                pre_img = s1_pre_patch
                post_img = s1_post_patch
            elif self.opt_only:
                s2_channels = s2_pre_patch.shape[-1]
                pre_img = s2_pre_patch
                post_img = s2_post_patch
            else:
                s1_channels = s1_pre_patch.shape[-1]
                s2_channels = s2_pre_patch.shape[-1]
                
                pre_img = np.concatenate([s2_pre_patch, s1_pre_patch], axis=-1)
                post_img = np.concatenate([s2_post_patch, s1_post_patch], axis=-1)

            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

            if not self.sar_only and not self.opt_only:
                s2_pre_patch = pre_img[..., :s2_channels]
                s1_pre_patch = pre_img[..., s2_channels:]
                s2_post_patch = post_img[..., :s2_channels]
                s1_post_patch = post_img[..., s2_channels:]
            elif self.sar_only:
                s1_pre_patch = pre_img
                s1_post_patch = post_img
            else:  # opt_only
                s2_pre_patch = pre_img
                s2_post_patch = post_img

        # Transpose the patches that exist
        if s1_pre_patch is not None:
            s1_pre_patch = np.transpose(s1_pre_patch, (2, 0, 1))
            s1_post_patch = np.transpose(s1_post_patch, (2, 0, 1))
        if s2_pre_patch is not None:
            s2_pre_patch = np.transpose(s2_pre_patch, (2, 0, 1))
            s2_post_patch = np.transpose(s2_post_patch, (2, 0, 1))

        return s1_pre_patch, s1_post_patch, s2_pre_patch, s2_post_patch, label

    

    def process_all_images(self):
        processed_data = {}
        for city in self.cities:
            data = {}
            
            if not self.opt_only:
                data['s1_pre'] = self.process_SAR(self.load_image(os.path.join(self.dataset_path, 'S1', city, 'imgs_1', 'transformed')))
                data['s1_post'] = self.process_SAR(self.load_image(os.path.join(self.dataset_path, 'S1', city, 'imgs_2', 'transformed')))

                data['s1_pre'] = imutils.normalize_img_custom(data['s1_pre'])
                data['s1_post'] = imutils.normalize_img_custom(data['s1_post'])
            
            if not self.sar_only:
                if self.opt_bands == 4:
                    data['s2_pre'] = self.process_MS(self.load_and_stack_tiff(
                        os.path.join(self.dataset_path, 'S2', city, 'imgs_1_rect'),
                        select_list=['B02','B03','B04','B08']))
                    data['s2_post'] = self.process_MS(self.load_and_stack_tiff(
                        os.path.join(self.dataset_path, 'S2', city, 'imgs_2_rect'),
                        select_list=['B02','B03','B04','B08']))
                else:
                    data['s2_pre'] = self.process_MS(self.load_and_stack_tiff(os.path.join(self.dataset_path, 'S2', city, 'imgs_1_rect')))
                    data['s2_post'] = self.process_MS(self.load_and_stack_tiff(os.path.join(self.dataset_path, 'S2', city, 'imgs_2_rect')))
                data['s2_pre'] = imutils.normalize_img_custom(data['s2_pre'])
                data['s2_post'] = imutils.normalize_img_custom(data['s2_post'])

            label_folder = f'{self.type.capitalize()} Labels'
            label_path = os.path.join(self.dataset_path, label_folder, city, 'cm')
            label_file = next(f for f in os.listdir(label_path) if f.endswith('-cm.tif'))
            data['label'] = imageio.imread(os.path.join(label_path, label_file)) - 1

            # Normalize
            
            

            processed_data[city] = data
            
        return processed_data 
    
    def calculate_class_weights(self):
        total_pixels = 0
        changed_pixels = 0
        for city in self.cities:
            label = self.processed_data[city]['label']
            total_pixels += label.size
            changed_pixels += np.sum(label == 1)  # Assuming 2 represents changed pixels

        unchanged_pixels = total_pixels - changed_pixels
        weights = [self.fp_modifier * 2 * changed_pixels / total_pixels, 2 * unchanged_pixels / total_pixels]
        return torch.tensor(weights, dtype=torch.float32)

    def __getitem__(self, index):
        patch_info = self.patches_info[index]
        city, top, left = patch_info['city'], patch_info['top'], patch_info['left']

        city_data = self.processed_data[city]
        h, w = self.patch_size, self.patch_size

        # Initialize variables
        s1_pre_patch = s1_post_patch = s2_pre_patch = s2_post_patch = None

        if not self.opt_only:
            s1_pre_patch = city_data['s1_pre'][top:top+h, left:left+w]
            s1_post_patch = city_data['s1_post'][top:top+h, left:left+w]

        if not self.sar_only:
            s2_pre_patch = city_data['s2_pre'][top:top+h, left:left+w]
            s2_post_patch = city_data['s2_post'][top:top+h, left:left+w]

        label_patch = city_data['label'][top:top+h, left:left+w]

        aug = self.type == 'train'
        s1_pre_patch, s1_post_patch, s2_pre_patch, s2_post_patch, label_patch = self.transforms(
            aug, s1_pre_patch, s1_post_patch, s2_pre_patch, s2_post_patch, label_patch)

        if self.sar_only:
            return s1_pre_patch, s1_post_patch, label_patch, city, top, left
        elif self.opt_only:
            return s2_pre_patch, s2_post_patch, label_patch, city, top, left
        else:
            return s1_pre_patch, s1_post_patch, s2_pre_patch, s2_post_patch, label_patch, city, top, left
        

    def __len__(self):
        return len(self.patches_info)


    def stitch_patches(self, patches, city):
        h, w = self.processed_data[city]['label'].shape[:2]
        reconstructed = np.zeros((h, w), dtype=np.float32)

        city_patches = [p for p in self.patches_info if p['city'] == city]

        for patch, patch_info in zip(patches, city_patches):
            i, j = patch_info['top'], patch_info['left']
            reconstructed[i:i+self.patch_size, j:j+self.patch_size] += patch
            reconstructed[reconstructed>0] = 1

        return reconstructed

    def stitch_all_cities(self, all_predictions):
        reconstructed_cities = {}
        predictions_by_city = defaultdict(list)

        for prediction, patch_info in zip(all_predictions, self.patches_info):
            predictions_by_city[patch_info['city']].append(prediction)

        for city in self.cities:
            city_predictions = predictions_by_city[city]
            reconstructed_cities[city] = self.stitch_patches(city_predictions, city)

        return reconstructed_cities



def make_data_loader(args, shuffle=True,  split = 'train', **kwargs):  # **kwargs could be omitted


    if 'ONERA-multimodal' in args.dataset:
        dataset = ONERAChangeDetectionDataset(args.dataset_path, type=split, patch_size=args.patch_size, stride=int(args.patch_size/8-1), 
                                              sar_only=args.sar_only, opt_only=args.opt_only, opt_bands=args.opt_bands)
        print('Dataset size:', len(dataset), ', type:', dataset.type)
        print('weights:', dataset.weights)
        if split=='train':
          data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs, num_workers=12,
                                  drop_last=False)
        else: 
          data_loader = DataLoader(dataset, batch_size=1, shuffle=False, **kwargs, 
                                  drop_last=False)
        return data_loader, dataset.weights

    else:
        raise NotImplementedError
def save_dataset_samples(dataset_path, save_dir):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get first sample from dataset
    train_ds = ONERAChangeDetectionDataset(dataset_path, type='train', opt_bands=4)
    sample = train_ds[200]
    
    # Unpack sample
    s1_pre_patch, s1_post_patch, s2_pre_patch, s2_post_patch, label_patch, city, top, left = sample
    
    # Dictionary to map array position to descriptive name
    name_mapping = {
        0: 'sar_pre',
        1: 'sar_post',
        2: 'optical_pre',
        3: 'optical_post',
        4: 'label'
    }
    
    # Save each array with descriptive name
    for idx, item in enumerate(sample):
        if isinstance(item, np.ndarray):
            # Create descriptive filename
            filename = f"{name_mapping[idx]}_{city}_{top}_{left}.png"
            
            # Process array for saving
            if idx < 4:  # Image data
                if idx in [0, 1]:  # SAR data
                    # Take only first channel and keep as grayscale
                    if item.shape[0] == 2:
                        item = item[0]  # Take first channel if channels-first
                    else:
                        item = item[..., 0]  # Take first channel if channels-last
                    
                    # Scale to 0-255 range for visualization
                    save_img = ((item - item.min()) * (255 / (item.max() - item.min()))).astype(np.uint8)
                    
                else:  # Optical data
                    # Scale to 0-255 range for visualization
                    scaled_img = ((item - item.min()) * (255 / (item.max() - item.min()))).astype(np.uint8)
                    
                    # Handle optical data channels
                    if scaled_img.ndim == 3:
                        if scaled_img.shape[0] >= 3:  # If channels-first
                            save_img = np.moveaxis(scaled_img[:3], 0, -1)
                        else:  # If channels-last
                            save_img = scaled_img[..., :3]
                    else:
                        save_img = np.repeat(scaled_img[..., np.newaxis], 3, axis=-1)
            
            else:  # Label data
                # For labels, create visualization
                save_img = (item * 255).astype(np.uint8)
            
            # Save the image
            save_path = os.path.join(save_dir, filename)
            imageio.imwrite(save_path, save_img)
            print(f"Saved {filename}")
        else:
            print(f"Skipped non-array item: {item}")

# Usage


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
    parser.add_argument('--dataset', type=str, default='WHUBCD')
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/STCD/data/ST-WHU-BCD')
    parser.add_argument('--data_list_path', type=str, default='./ST-WHU-BCD/train_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_name_list', type=list)

    args = parser.parse_args()

   
    save_dir = r"C:\GridEyeS\MambaCD\changedetection\figures"
        
    dataset_path = r"C:\GridEyeS\Data\ONERA"
    save_dataset_samples(dataset_path, save_dir)

