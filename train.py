import torch
import time
import argparse
import os
import numpy as np
import cv2
from model import fusion_refine
from train_dataset import dehaze_train_dataset
from test_dataset import dehaze_test_dataset
from val_dataset import dehaze_val_dataset
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.models import vgg16
from utils_test import psnr, ssim
import torch.nn.functional as F
import torch.nn as nn
from perceptual import LossNetwork
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim
from retinex import retinex_lowlight_fix, retinex_tensor
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalars(self, *args, **kwargs):
            pass
        def close(self):
            pass

class SobelXY(nn.Module):
    def __init__(self):
        super(SobelXY, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)
        
    def forward(self,x):
        b,c,h,w=x.shape
        weightx = self.weightx.repeat(c,1,1,1)
        weighty = self.weighty.repeat(c,1,1,1)
        sobel_x=F.conv2d(x, weightx, groups=c, padding=1)
        sobel_y=F.conv2d(x, weighty, groups=c, padding=1)
        return torch.abs(sobel_x)+torch.abs(sobel_y)

class L1_Sobel_Loss(nn.Module):
    def __init__(self):
        super(L1_Sobel_Loss, self).__init__()
        self.sobel = SobelXY()
        self.l1 = nn.L1Loss()
    def forward(self, x, y):
        return self.l1(self.sobel(x), self.sobel(y))

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps * self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


def compute_retinex_reflection(clear_tensor):
    return retinex_tensor(clear_tensor)


def rgb_color_only(x):
    y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return x - y.expand_as(x)


def split_dataset_names(names):
    return [name.strip() for name in names.split(',') if name.strip()]


def find_pair_dir(data_dir, split, name):
    candidates = [
        os.path.join(data_dir, split, name),
        os.path.join(data_dir, name, split),
        os.path.join(data_dir, name),
    ]
    pair_dir_names = [
        ('haze', 'clear'),
        ('hazy', 'GT'),
        ('hazy', 'gt'),
        ('hazy', 'clear'),
    ]
    for candidate in candidates:
        for haze_name, clean_name in pair_dir_names:
            if os.path.isdir(os.path.join(candidate, haze_name)) and os.path.isdir(os.path.join(candidate, clean_name)):
                return candidate
    raise FileNotFoundError(
        'Cannot find paired dataset for "{}" split "{}". Tried: {}'.format(
            name, split, ', '.join(candidates)
        )
    )


def build_concat_dataset(dataset_cls, data_dir, split, names):
    datasets = []
    for name in split_dataset_names(names):
        dataset_dir = find_pair_dir(data_dir, split, name)
        dataset = dataset_cls(dataset_dir)
        if len(dataset) == 0:
            raise RuntimeError('Dataset "{}" has no paired images: {}'.format(name, dataset_dir))
        print('Loaded {} {} images from {}'.format(len(dataset), name, dataset_dir))
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

parser = argparse.ArgumentParser(description='RCAN-Dehaze-teacher')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=20, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=10000, type=int)
parser.add_argument('--train_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='..')
parser.add_argument('--train_sets', type=str, default='fog', help='Comma-separated dataset names, e.g. fog,dark')
parser.add_argument('--test_sets', type=str, default='', help='Comma-separated test dataset names. Defaults to --train_sets')
parser.add_argument('--model_save_dir', type=str, default='./output_result')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--test_log_file', type=str, default=None)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1,  type=int)
parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
parser.add_argument('--imagenet_model', default='', type=str, help='load trained model or not')
parser.add_argument('--rcan_model', default='', type=str, help='load trained model or not')
parser.add_argument('--resume_model', default='', type=str, help='Path to a checkpoint to continue training from')
parser.add_argument('--start_epoch', default=0, type=int, help='First epoch index when resuming training')
args = parser.parse_args()
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch
train_sets = args.train_sets
test_sets = args.test_sets if args.test_sets else args.train_sets
start_epoch = args.start_epoch

# --- test --- #
val_dataset = os.path.join(args.data_dir, 'val')
predict_result= args.predict_result
test_batch_size=args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir=os.path.join(args.model_save_dir,'output_result')
if args.test_log_file is None:
    test_log_path = os.path.join(args.model_save_dir, 'test_log.txt')
else:
    test_log_path = args.test_log_file

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
MyEnsembleNet = fusion_refine(args.imagenet_model, args.rcan_model)
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=learning_rate)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
    G_optimizer,
    milestones=[50, 70, 80],
    gamma=0.5
)
# --- Load training data --- #
dataset = build_concat_dataset(dehaze_train_dataset, args.data_dir, 'train', train_sets)
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True)
# --- Load testing data --- #
test_dataset = build_concat_dataset(dehaze_test_dataset, args.data_dir, 'test', test_sets)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

val_dataset = dehaze_val_dataset(val_dataset)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
MyEnsembleNet = MyEnsembleNet.to(device)
if len(device_ids) > 1:
    MyEnsembleNet = torch.nn.DataParallel(MyEnsembleNet, device_ids=device_ids)
writer = SummaryWriter(os.path.join(args.model_save_dir, 'tensorboard'))

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True)
# vgg_model.load_state_dict(torch.load(os.path.join(args.vgg_model , 'vgg16.pth')))
vgg_model = vgg_model.features[:16].to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()

msssim_loss = msssim
sobel_loss_func = L1_Sobel_Loss().to(device)
charbonnier_loss = CharbonnierLoss().to(device)
reflect_l1 = nn.L1Loss().to(device)

# '''server vgg'''
# vgg_model = vgg16(pretrained=True).features[:16]
# vgg_model = vgg_model.to(device)
# for param in vgg_model.parameters():
#     param.requires_grad = False
# loss_network = LossNetwork(vgg_model)
# loss_network.eval()


def load_model_weight(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    model_is_parallel = isinstance(model, torch.nn.DataParallel)
    state_is_parallel = any(k.startswith('module.') for k in state.keys())

    if state_is_parallel and not model_is_parallel:
        state = {k[len('module.'):]: v for k, v in state.items()}
    elif model_is_parallel and not state_is_parallel:
        state = {'module.' + k: v for k, v in state.items()}

    model.load_state_dict(state)


# --- Load the network weight --- #
if args.resume_model:
    load_model_weight(MyEnsembleNet, args.resume_model, device)
    print('--- resumed weight from {} ---'.format(args.resume_model))
else:
    print('--- no weight loaded ---')

# --- Strat training --- #
iteration = 0
epoch_total_losses = []
epoch_char_losses = []
epoch_perc_losses = []
epoch_msssim_losses = []
epoch_sobel_losses = []
epoch_reflect_losses = []

loss_log_path = os.path.join(args.model_save_dir, "loss_epoch_log.txt")
if start_epoch == 0 or not os.path.exists(loss_log_path):
    with open(loss_log_path, "w", encoding="utf-8") as f:
        f.write("epoch\ttotal\tchar\tperceptual\tmsssim\tsobel\treflect\n")

if start_epoch > 0:
    lr_scale = scheduler_G.gamma ** sum(start_epoch >= milestone for milestone in scheduler_G.milestones)
    for group in G_optimizer.param_groups:
        group['lr'] = learning_rate * lr_scale
    scheduler_G.last_epoch = start_epoch
    print('--- resume from epoch {}, learning rate set to {:.8f} ---'.format(
        start_epoch, G_optimizer.param_groups[0]['lr']
    ))

for epoch in range(start_epoch, train_epoch):
    start_time = time.time()
    MyEnsembleNet.train()
    print(epoch)
    epoch_total = 0.0
    epoch_char = 0.0
    epoch_perc = 0.0
    epoch_msssim = 0.0
    epoch_sobel = 0.0
    epoch_reflect = 0.0
    batch_count = 0
    for batch_idx, (haze, clear) in enumerate(train_loader):
        # print(batch_idx)
        iteration +=1
        haze = haze.to(device)
        clear = clear.to(device)
        output, rcan_reflect = MyEnsembleNet(haze)

        MyEnsembleNet.zero_grad()
        # smooth_loss_l1 = F.smooth_l1_loss(output, clear)
        loss_char = charbonnier_loss(output, clear)
        perceptual_loss = loss_network(output, clear)
        msssim_loss_ = 1 - msssim_loss(output, clear, val_range=1.0, normalize=True)
        sobel_loss = sobel_loss_func(output, clear)
        retinex_reflect = compute_retinex_reflection(clear)
        rcan_reflect = compute_retinex_reflection(rcan_reflect)
        reflect_loss = reflect_l1(rcan_reflect, retinex_reflect)
        total_loss = (
            loss_char
            + 0.1 * perceptual_loss
            + 0.5 * msssim_loss_
            + 0.1 * sobel_loss
            + 0.01 * reflect_loss
        )

        total_loss.backward()
        G_optimizer.step()

        epoch_total += float(total_loss.item())
        epoch_char += float(loss_char.item())
        epoch_perc += float(perceptual_loss.item())
        epoch_msssim += float(msssim_loss_.item())
        epoch_sobel += float(sobel_loss.item())
        epoch_reflect += float(reflect_loss.item())
        batch_count += 1

        print(f'epoch {epoch} step {batch_idx}/{len(train_loader)} '
              f'total:{total_loss.item():.4f} '
              f'char:{loss_char.item():.4f} '
              f'perc:{perceptual_loss.item():.4f} '
              f'msssim:{msssim_loss_.item():.4f} '
              f'sobel:{sobel_loss.item():.4f} '
              f'refl:{reflect_loss.item():.4f} ')
#         if iteration % 2 == 0:
#             frame_debug = torch.cat(
#                 (hazy, output, clean), dim=0)
#             writer.add_images('train_debug_img', frame_debug, iteration)
        writer.add_scalars('training', {'training total loss': total_loss.item()
                                        }, iteration)
        writer.add_scalars('training_img', {
                                            'img loss_char': loss_char.item(),
                                            'perceptual': perceptual_loss.item(),
                                            'msssim': msssim_loss_.item(),
                                            'sobel': sobel_loss.item(),
                                            'reflect': reflect_loss.item()
                                            }, iteration)

    if batch_count > 0:
        avg_total = epoch_total / batch_count
        avg_char = epoch_char / batch_count
        avg_perc = epoch_perc / batch_count
        avg_msssim = epoch_msssim / batch_count
        avg_sobel = epoch_sobel / batch_count
        avg_reflect = epoch_reflect / batch_count
    else:
        avg_total = avg_char = avg_perc = avg_msssim = avg_sobel = avg_reflect = 0.0

    epoch_total_losses.append(avg_total)
    epoch_char_losses.append(avg_char)
    epoch_perc_losses.append(avg_perc)
    epoch_msssim_losses.append(avg_msssim)
    epoch_sobel_losses.append(avg_sobel)
    epoch_reflect_losses.append(avg_reflect)

    with open(loss_log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch}\t{avg_total:.6f}\t{avg_char:.6f}\t{avg_perc:.6f}\t"
            f"{avg_msssim:.6f}\t{avg_sobel:.6f}\t{avg_reflect:.6f}\n"
        )

    if epoch % 20 == 0:
        print('we are testing on epoch: ' + str(epoch))
        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            recon_psnr_list = []
            recon_ssim_list = []
            MyEnsembleNet.eval()
            for batch_idx, batch in enumerate(test_loader):
                if len(batch) == 5:
                    image_name, _, _, haze, clear = batch
                elif len(batch) == 4:
                    image_name, _, haze, clear = batch
                else:
                    raise ValueError('Unexpected test batch format with {} fields'.format(len(batch)))
                # hazy_up = hazy_up.to(device)
                # hazy_down=hazy_down.to(device)
                clear = clear.to(device)
                haze = haze.to(device)
                frame_out, _ = MyEnsembleNet(haze)
                # frame_out_up = MyEnsembleNet(hazy_up)
                # frame_out_down = MyEnsembleNet(hazy_down)
                # frame_out=(torch.cat([frame_out_up.permute(0,2,3,1), frame_out_down[:,:,80:640,:].permute(0,2,3,1)], 1)).permute(0,3,1,2)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
#                 imwrite(frame_out, output_dir +'/' +str(batch_idx) + '.png', range=(0, 1))

                psnr_list.append(psnr(frame_out, clear))
                ssim_list.append(ssim(frame_out, clear))



            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)
            print(epoch,'dehazed', avr_psnr, avr_ssim)
            try:
                with open(test_log_path, 'a', encoding='utf-8') as f:
                    f.write(f'{epoch}\t{float(avr_psnr):.4f}\t{float(avr_ssim):.4f}\n')
            except Exception as e:
                print(f'write test log failed: {e}')
            writer.add_scalars('testing', {'testing psnr':avr_psnr,
                'testing ssim': avr_ssim
                                    }, epoch)
            torch.save(MyEnsembleNet.state_dict(), os.path.join(args.model_save_dir,'epoch'+ str(epoch) + '.pkl'))

    scheduler_G.step()

epochs_range = range(len(epoch_total_losses))
plt.figure()
plt.plot(epochs_range, epoch_total_losses, label="total")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.model_save_dir, "loss_total.png"))
plt.close()

plt.figure()
plt.plot(epochs_range, epoch_char_losses, label="char")
plt.plot(epochs_range, epoch_perc_losses, label="perceptual")
plt.plot(epochs_range, epoch_msssim_losses, label="msssim")
plt.plot(epochs_range, epoch_sobel_losses, label="sobel")
plt.plot(epochs_range, epoch_reflect_losses, label="reflect")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.model_save_dir, "loss_components.png"))
plt.close()

writer.close()
