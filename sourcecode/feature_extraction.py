import os
import glob
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import kurtosis, skew
from tqdm import tqdm

def extract_rgb_his_features_paper(image_path, resize=None):

    img = cv2.imread(image_path)
    if img is None:
        return None, None
    if resize is not None:
        img = cv2.resize(img, resize)

    B = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    R = img[:, :, 2].astype(np.float32)

    denom = R + G + B + 1e-6
    Rn = R / denom
    Gn = G / denom
    Bn = B / denom

    I = (Rn + Gn + Bn) / 3.0
    S = 1.0 - (3.0 / (Rn + Gn + Bn + 1e-6)) * np.minimum(np.minimum(Rn, Gn), Bn)

    num = 0.5 * ((Rn - Gn) + (Rn - Bn))
    den = np.sqrt((Rn - Gn)**2 + (Rn - Bn)*(Gn - Bn)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    theta_deg = np.degrees(theta)
    H = np.where(Bn <= Gn, theta_deg, 360.0 - theta_deg)

    def stats(x):
        return [np.mean(x), np.var(x), np.max(x) - np.min(x)]

    rgb_feats = stats(R) + stats(G) + stats(B)
    his_feats = stats(H) + stats(I) + stats(S)
    return rgb_feats, his_feats


def extract_rgb_his(image_dirs, rgb_csv, his_csv, dataset, resize=None):
    print(f"Processing '{dataset}' (RGB+HIS)...")
    rgb_data, his_data = [], []
    for folder in image_dirs:
        label = os.path.basename(folder)
        for path in tqdm(glob.glob(os.path.join(folder, '*.png')), desc=label):
            rgb_feat, his_feat = extract_rgb_his_features_paper(path, resize)
            if rgb_feat is None:
                print(f"Error reading {path}")
                continue
            fn = os.path.basename(path)
            rgb_data.append([fn, label] + rgb_feat)
            his_data.append([fn, label] + his_feat)

    rgb_cols = ['filename', 'label',
                'mean_R', 'var_R', 'range_R',
                'mean_G', 'var_G', 'range_G',
                'mean_B', 'var_B', 'range_B']
    his_cols = ['filename', 'label',
                'mean_H', 'var_H', 'range_H',
                'mean_I', 'var_I', 'range_I',
                'mean_S', 'var_S', 'range_S']

    pd.DataFrame(rgb_data, columns=rgb_cols).to_csv(rgb_csv, index=False)
    pd.DataFrame(his_data, columns=his_cols).to_csv(his_csv, index=False)
    print(f"RGB values for {dataset} saved in {rgb_csv}, HIS values for {dataset} saved in {his_csv}")


def extract_lab(image_dirs, save_csv, dataset, resize=None):
    print(f"Processing '{dataset}' (Lab)...")
    data = []
    for folder in image_dirs:
        label = os.path.basename(folder)
        for path in tqdm(glob.glob(os.path.join(folder, '*.png')), desc=label):
            img = cv2.imread(path)
            if img is None:
                print(f" Error reading {path}")
                continue
                
            img = img.astype(np.float32) / 255.0
            
            if resize is not None:
                img = cv2.resize(img, resize)
                
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            L, A, B = cv2.split(lab)

            stats_fns = [np.mean, np.var, lambda x: np.max(x) - np.min(x)]
            row = [os.path.basename(path), label]
            for channel in (L, A, B):
                row += [fn(channel) for fn in stats_fns]
            data.append(row)

    cols = ['filename', 'label',
            'L_mean', 'L_var', 'L_range',
            'A_mean', 'A_var', 'A_range',
            'B_mean', 'B_var', 'B_range']
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(save_csv, index=False)
    print(f"LAB values for {dataset} saved in {save_csv}")



import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import kurtosis, skew

def extract_glcm(image_dirs, save_csv, dataset, resize=None, angles=[0]):
    print(f"Processing '{dataset}' (GLCM, angles={angles})...")
    data = []
    i_vals = np.arange(64)
    j_vals = np.arange(64)
    denom = 1 + (i_vals[:, None] - j_vals) ** 2 
    for folder in image_dirs:
        label = os.path.basename(folder)
        for path in tqdm(glob.glob(os.path.join(folder, '*.png')), desc=label):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error reading {path}")
                continue
            if resize is not None:
                img = cv2.resize(img, resize)
            img = (img // 4).astype(np.uint8)

            glcm = graycomatrix(
                img,
                distances=[1],
                angles=angles,
                levels=64,
                symmetric=True,
                normed=True
            )

            contrast_vals    = graycoprops(glcm, 'contrast').flatten() 
            correlation_vals = graycoprops(glcm, 'correlation').flatten()
            energy_vals      = graycoprops(glcm, 'energy').flatten()
            homogeneity_vals = graycoprops(glcm, 'homogeneity').flatten()

            contrast_avg    = np.mean(contrast_vals)
            correlation_avg = np.mean(correlation_vals)
            energy_avg      = np.mean(energy_vals)
            homogeneity_avg = np.mean(homogeneity_vals)

            angle_count = len(angles)
            idm_vals = []
            for k in range(angle_count):
                P = glcm[:, :, 0, k]  
                idm_k = np.sum(P / denom)
                idm_vals.append(idm_k)
            idm_avg = np.mean(idm_vals)
     
            entropy_vals = []
            for k in range(angle_count):
                P = glcm[:, :, 0, k]
                entropy_k = -np.sum(P * np.log2(P + 1e-10))
                entropy_vals.append(entropy_k)
            entropy_avg = np.mean(entropy_vals)

            feats = {
                'contrast':    contrast_avg,
                'correlation': correlation_avg,
                'energy':      energy_avg,
                'homogeneity': homogeneity_avg,
                'mean':        np.mean(img),
                'std':         np.std(img),
                'entropy':     entropy_avg,
                'rms':         np.sqrt(np.mean(img**2)),
                'variance':    np.var(img),
                'smoothness':  1 - 1 / (1 + np.var(img)),
                'kurtosis':    kurtosis(img.flatten()),
                'skewness':    skew(img.flatten()),
                'idm':         idm_avg
            }
            feats['filename'] = os.path.basename(path)
            feats['label']    = label
            data.append(feats)

    df = pd.DataFrame(data)
    cols = ['filename', 'label'] + [c for c in df.columns if c not in ('filename', 'label')]
    df = df[cols]
    df.to_csv(save_csv, index=False)
    print(f"GLCM values for {dataset} (angles={angles}) saved in {save_csv}")

def extract_lbp(image_dirs, save_csv, dataset, P=8, R=1, resize=None):
    print(f"Processing '{dataset}' (LBP)...")
    data = []
    n_bins = P * (P - 1) + 3
    for folder in image_dirs:
        label = os.path.basename(folder)
        for path in tqdm(glob.glob(os.path.join(folder, '*.png')), desc=label):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error reading {path}")
                continue
            if resize is not None:
                img = cv2.resize(img, resize)
            lbp = local_binary_pattern(img, P, R, method='nri_uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(float) / (hist.sum() + 1e-6)
            row = {'filename': os.path.basename(path), 'label': label}
            row.update({f'lbp_{i}': v for i,v in enumerate(hist)})
            data.append(row)

    df = pd.DataFrame(data)
    cols = ['filename','label'] + [c for c in df.columns if c not in ('filename','label')]
    df = df[cols]
    df.to_csv(save_csv, index=False)
    print(f"LBP values for {dataset} saved in {save_csv}")


