import torch
import time
import argparse
import numpy as np
from test_dataset import dehaze_test_dataset
from torch.utils.data import DataLoader
import os
from utils_test import psnr, ssim
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from torch.cuda.amp import autocast

# --- Parse hyper-parameters test --- #
parser = argparse.ArgumentParser(description='RCAN-Dehaze-test')
parser.add_argument('--data_dir', type=str, default='../test/fog')
parser.add_argument('--model_save_dir', type=str, default='./results')
parser.add_argument('--model_file', type=str, default='./output_result/epoch100.pkl')
parser.add_argument('--imagenet_model', default='', type=str, help='load trained model or not')
parser.add_argument('--rcan_model', default='', type=str, help='load trained model or not')
parser.add_argument('--model_version', default='current', choices=['current', 'original50'],
                    help='Model definition used by the checkpoint')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
args = parser.parse_args()


def strip_module_prefix(state):
    if any(k.startswith('module.') for k in state.keys()):
        return {k[len('module.'):]: v for k, v in state.items()}
    return state


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    return strip_module_prefix(state)


def select_fusion_refine(model_version):
    if model_version == 'original50':
        from model_original50 import fusion_refine
    else:
        from model import fusion_refine
    print('--- using {} model definition ---'.format(model_version))
    return fusion_refine

# --- output picture --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir = os.path.join(args.model_save_dir, '')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

checkpoint = load_checkpoint(args.model_file, device)
fusion_refine = select_fusion_refine(args.model_version)

# --- Define the network --- #
MyEnsembleNet = fusion_refine(args.imagenet_model, args.rcan_model)
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))

# --- Dataset --- #
val_dataset = dehaze_test_dataset(args.data_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU (optional) --- #
MyEnsembleNet = MyEnsembleNet.to(device)
if len(device_ids) > 1:
    MyEnsembleNet = torch.nn.DataParallel(MyEnsembleNet, device_ids=device_ids)

# --- Load the network weight --- #
# try:
#     if os.path.exists(args.model_file):
load_state = checkpoint
if isinstance(MyEnsembleNet, torch.nn.DataParallel):
    load_state = {'module.' + k: v for k, v in checkpoint.items()}
MyEnsembleNet.load_state_dict(load_state)
print(f'--- weight loaded from {args.model_file} ---')
#     elif os.path.exists('epoch95.pkl'):
#          MyEnsembleNet.load_state_dict(torch.load('epoch95.pkl'))
#          print('--- weight loaded from epoch95.pkl ---')
#     else:
#         print(f'--- no weight loaded (checked {args.model_file} and epoch95.pkl) ---')
# except Exception as e:
#     print(f'--- weight load failed: {e} ---')

# --- Start testing --- #
with torch.no_grad():
    time_list = []
    psnr_list = []
    ssim_list = []
    MyEnsembleNet.eval()

    for batch_idx, batch in enumerate(val_loader):
        if len(batch) == 5:
            image_name, _, _, haze, clear = batch
        elif len(batch) == 4:
            image_name, _, haze, clear = batch
        else:
            raise ValueError('Unexpected test batch format with {} fields'.format(len(batch)))
        haze_image = haze.to(device)
        clean_image = clear.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()

        if device.type == 'cuda':
            with autocast():
                img_tensor, rcan_reflect = MyEnsembleNet(haze_image)
        else:
            img_tensor, rcan_reflect = MyEnsembleNet(haze_image)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()

        time_list.append((end - start))

        # Save the image
        name_part = os.path.splitext(image_name[0])[0]
        save_path = os.path.join(output_dir, name_part + '.png')

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        output = img_tensor.clamp(0, 1).float()
        clean_image = clean_image.float()
        ts = torch.squeeze(output.cpu())
        vutils.save_image(ts, save_path)

        # Directly compute metrics without reload
        psnr_val = psnr(output, clean_image)
        ssim_val = ssim(output, clean_image).item()

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        print(f"  PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")

    if len(time_list) > 0:
        time_cost = float(sum(time_list) / len(time_list))
        print('running time per image: {:.4f}s'.format(time_cost))
        print('FPS: {:.2f}'.format(1 / time_cost))
        if psnr_list:
            print('Average PSNR: {:.4f}'.format(sum(psnr_list) / len(psnr_list)))
            print('Average SSIM: {:.4f}'.format(sum(ssim_list) / len(ssim_list)))
    else:
        print('No images processed.')
