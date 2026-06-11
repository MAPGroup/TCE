import torch
import torch.nn.functional as F
from math import log10
import cv2
import numpy as np
import torchvision
from skimage.metrics import structural_similarity as ssim
# def to_psnr(frame_out, gt):
#     mse = F.mse_loss(frame_out, gt, reduction='none')
#     mse_split = torch.split(mse, 1, dim=0)
#     mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
#     intensity_max = 1.0
#     psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
#     return psnr_list

# def to_ssim_skimage(dehaze, gt):
#     dehaze_list = torch.split(dehaze, 1, dim=0)
#     gt_list = torch.split(gt, 1, dim=0)

#     dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
#     gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
#     ssim_list = [ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, channel_axis=2) for ind in range(len(dehaze_list))]

#     return ssim_list



from math import exp
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from  torchvision.transforms import ToPILImage

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1=torch.clamp(img1,min=0,max=1)
    img2=torch.clamp(img2,min=0,max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(pred, gt):
    pred=pred.clamp(0,1).cpu().numpy()
    gt=gt.clamp(0,1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10( 1.0 / rmse)

if __name__ == "__main__":
    pass






def predict(gridnet, test_data_loader):

    psnr_list = []
    for batch_idx, (frame1, frame2, frame3) in enumerate(test_data_loader):
        with torch.no_grad():
            frame1 = frame1.to(torch.device('cuda'))
            frame3 = frame3.to(torch.device('cuda'))
            gt = frame2.to(torch.device('cuda'))
            # print(frame1)

            frame_out = gridnet(frame1, frame3)
            # print(frame_out)
            frame_debug = torch.cat((frame1, frame_out, gt, frame3), dim =0)
            filepath = "./image" + str(batch_idx) + '.png'
            torchvision.utils.save_image(frame_debug, filepath)
            # print(frame_out)
            # img = np.asarray(frame_out.cpu()).astype(float)
            
            # cv2.imwrite(filepath , img)



        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(frame_out, gt))
    avr_psnr = sum(psnr_list) / len(psnr_list)
    return avr_psnr


