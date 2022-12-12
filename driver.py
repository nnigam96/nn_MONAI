import torch
import argparse
from monai.data import ImageDataset
from monai.networks.nets import UNet
from utils import *
from constants import *
from Data import *
from model import *
import pprint
from datetime import date

# New comment added to check git push

def main():
    parser =  argparse.ArgumentParser(description='Driver for MRI denoising experiments using MONAI')
    parser.add_argument('-lr', help='Learning rate', default=1e-3)
    parser.add_argument('-epochs', help='Epochs for training', default=100)
    parser.add_argument('-dir_name', help='Directory with nii files', default=DATA_DIR)
    parser.add_argument('-model_tag', help='UID for saved model', default='UNet with Rand Crop')
    parser.add_argument('-continue_training', help='Resume from a known add', default=False)
    parser.add_argument('-callback_loc', help='Path to saved model for callback', default=r"/staging/nnigam/exp_results/UNet with Rand Crop_2022-11-30")
    
    args = parser.parse_args()

    filenames =  fetch_nii_file_names(args.dir_name)
    check_ds = ImageDataset(image_files=filenames, seg_files= filenames, transform=ip_train_transforms, seg_transform=label_train_transforms)
    train_loader, test_loader, val_loader = get_loaders(check_ds)
    

    args.model_tag = args.model_tag + '_'+str(date.today())
    
    Unet3D = UNet(  spatial_dims=3,
                    in_channels=1,
                    out_channels=1,
                    channels=(4, 8, 16, 32),
                    strides=(2, 2, 2),
                    num_res_units=8 
                )

    if args.continue_training:
        Unet3D = torch.load(args.callback_loc)

    best_model, train_loss, val_loss = model.train(Unet3D, train_loader, val_loader, args.lr, args.model_tag, args.epochs)
    model.loss_curves(args.model_tag, train_loss, val_loss)
    results = model.test(best_model, test_loader)
    model.infer(best_model, test_loader, args.model_tag)
    pprint.pprint(results)

if __name__ == '__main__':
    main()
