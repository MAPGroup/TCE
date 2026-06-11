import cv2
import numpy as np
import os
import torch

# --- 物理模型组件 ---

def get_lightness_score(img):
    """判断亮度：返回平均亮度值"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])

def get_haze_score(img):
    """判断雾气：利用暗通道的浓度来估计雾的程度"""
    img_norm = img.astype(np.float64) / 255.0
    dark = np.min(img_norm, axis=2)
    return np.mean(dark)

def retinex_lowlight_fix(img):
    """
    改进的低光照增强 (基于光照图估计 + Gamma矫正)
    替代旧的 SSR 方法，减少色彩失真和噪声
    """
    # 归一化 [0,1]
    img_norm = img.astype(np.float64) / 255.0
    
    # 1. 估计初始光照图 L_init (Max-RGB)
    # L(x) = max_{c} I_c(x)
    L_init = np.max(img_norm, axis=2)
    
    # 2. 构造引导图
    # 使用原图的灰度图作为引导图，保留边缘纹理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    
    # 3. 优化光照图 (使用引导滤波)
    # r=16, eps=1e-3
    L_refined = guided_filter(gray, L_init, r=16, eps=1e-3)
    
    # 4. Gamma 矫正提亮
    # Gamma < 1.0 提亮暗部
    gamma = 0.5 
    L_enhanced = np.power(np.maximum(L_refined, 0.001), gamma)
    
    # 5. 图像还原
    # J = I * (L_enhanced / L_refined)
    # 增加一个极小值防止除零
    ratio = L_enhanced / np.maximum(L_refined, 0.001)
    
    # 限制增益上限，防止噪声过度放大
    ratio = np.clip(ratio, 1.0, 5.0)
    
    # 广播增益到 3 通道
    img_enhanced = np.zeros_like(img_norm)
    for i in range(3):
        img_enhanced[:,:,i] = img_norm[:,:,i] * ratio
        
    return np.clip(img_enhanced * 255, 0, 255).astype(np.uint8)

def guided_filter(I, p, r, eps):
    """
    引导滤波 (Guided Filter) 实现
    I: 引导图像 (Guide Image)
    p: 输入图像 (Input Image)
    r: 滤波半径
    eps: 正则化参数
    """
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    return mean_a * I + mean_b

def dark_channel_dehaze(img):
    """暗通道先验物理去雾（优化版：带大气光估计和引导滤波）"""
    img_norm = img.astype(np.float64) / 255.0
    
    # 1. 计算暗通道
    patch_size = 15
    dark = np.min(cv2.erode(img_norm, np.ones((patch_size, patch_size))), axis=2)
    
    # 2. 估计大气光 A
    # 取暗通道中最亮的前 0.1% 的像素
    flat_dark = dark.flatten()
    num_pixels = flat_dark.size
    num_top = int(max(num_pixels * 0.001, 1))
    
    indices = np.argpartition(flat_dark, -num_top)[-num_top:]
    
    # 在原图中找到这些位置对应的像素，取最亮的作为大气光 A
    # 这里简单取 RGB 的平均值，或者取最大值
    # 更稳健的方法：取这些像素中亮度(Max RGB)最高的那个像素的 RGB 值
    
    img_flat = img_norm.reshape(-1, 3)
    top_pixels = img_flat[indices]
    
    # 计算这些像素的强度
    top_intensities = np.max(top_pixels, axis=1)
    max_idx = np.argmax(top_intensities)
    A = top_pixels[max_idx] # RGB 向量
    
    # 限制 A 的范围，防止过亮导致图像变黑
    A = np.clip(A, 0.1, 1.0)
    
    # 3. 估计透射率 t
    # t = 1 - omega * min(I / A)
    # 此时需要对每个通道分别除以 A
    norm_I = img_norm / A
    dark_t = np.min(cv2.erode(norm_I, np.ones((patch_size, patch_size))), axis=2)
    t_raw = 1.0 - 0.95 * dark_t
    
    # 4. 优化透射率 (使用引导滤波)
    # 使用原图的灰度图作为引导图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    t_refined = guided_filter(gray, t_raw, r=60, eps=1e-6)
    
    # 限制 t 的下界
    t_final = np.maximum(t_refined, 0.1)
    
    # 5. 恢复图像
    # J = (I - A) / t + A
    out = np.zeros_like(img_norm)
    for i in range(3):
        out[:,:,i] = (img_norm[:,:,i] - A[i]) / t_final + A[i]
        
    # 6. 曝光补偿 (可选，防止去雾后过暗)
    # out = out ** 0.8 
    
    return np.clip(out * 255, 0, 255).astype(np.uint8)

# --- 核心：自动判别系统 ---

def auto_restoration_system(image_path_or_img):
    if isinstance(image_path_or_img, str):
        img = cv2.imread(image_path_or_img)
    else:
        img = image_path_or_img
    if img is None:
        return

    l_score = get_lightness_score(img)
    h_score = get_haze_score(img)

    # print(f"检测指标 -> 亮度: {l_score:.2f}, 雾度: {h_score:.2f}")

    if l_score < 70:
        # print("系统判断：检测到【暗光退化】，启动 Retinex 增强...")
        processed = retinex_lowlight_fix(img)
    elif h_score > 0.4:
        # print("系统判断：检测到【雾天退化】，启动 DCP 去雾...")
        processed = dark_channel_dehaze(img)
    else:
        # print("系统判断：图像质量良好，执行轻微对比度优化...")
        processed = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

    return processed


def retinex_tensor(image_tensor):
    image_np = image_tensor.detach().cpu().numpy()
    b, c, h, w = image_np.shape
    out_list = []
    for i in range(b):
        img = image_np[i].transpose(1, 2, 0)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ref_bgr = auto_restoration_system(img_bgr)
        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        ref = ref_rgb.astype(np.float32) / 255.0
        out_list.append(ref.transpose(2, 0, 1))
    out_np = np.stack(out_list, axis=0)
    return torch.from_numpy(out_np).to(image_tensor.device)

