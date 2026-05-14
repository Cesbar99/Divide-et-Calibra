import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.utils import *
from scipy.io import loadmat
import medmnist
from medmnist import TissueMNIST, PathMNIST, INFO
from torch.utils.data import random_split
import os
from tiny_imagenet_torch import TinyImageNet
from datasets import load_dataset
from PIL import Image
import io
import numpy as np





def generateCalibrationData(kwargs, dataname=None):
    #temperature = str(int(kwargs.checkpoint.temperature))

    if kwargs.data == 'synthetic':   
        test_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
        'pre-train',
        kwargs.data,
        kwargs.checkpoint.num_classes,
        kwargs.checkpoint.num_features,
        kwargs.checkpoint.seed,
        kwargs.checkpoint.epochs,
        kwargs.checkpoint.temperature       
        )
        cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            'pre-train',
            kwargs.data,
            kwargs.checkpoint.num_classes,
            kwargs.checkpoint.num_features,
            kwargs.checkpoint.seed,
            kwargs.checkpoint.epochs,
            kwargs.checkpoint.temperature
        )           
    else:        
        test_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            'pre-train',
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            kwargs.checkpoint.seed,
            kwargs.checkpoint.epochs,
            kwargs.checkpoint.temperature       
        )
        cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            'pre-train',
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            kwargs.checkpoint.seed,
            kwargs.checkpoint.epochs,
            kwargs.checkpoint.temperature
        )    
    
    # Load your data
    df_train_calibration_data = pd.read_csv(test_results)
    df_eval_calibration_data = pd.read_csv(cal_results)    

    # Extract features and labels
    X_train_cal = df_train_calibration_data.drop(columns=["true", "preds"]).values
    std_per_column = np.std(X_train_cal, axis=0)
    print("Standard deviation per column:")
    print(std_per_column)
    print("Mean standard deviation per column:")
    print(np.mean(std_per_column))
    
    y_train_cal = df_train_calibration_data["true"].values
    p_train_cal = df_train_calibration_data["preds"].values

    X_eval_cal_full = df_eval_calibration_data.drop(columns=["true", "preds"]).values
    y_eval_cal_full = df_eval_calibration_data["true"].values
    p_eval_cal_full = df_eval_calibration_data["preds"].values

    # Split into 90% test and 10% val
    X_test_cal, X_val_cal, y_test_cal, y_val_cal, p_test_cal, p_val_cal = train_test_split(X_eval_cal_full,
                                                                                            y_eval_cal_full,
                                                                                            p_eval_cal_full,
                                                                                            test_size=0.1,
                                                                                            random_state=kwargs.seed,
                                                                                            stratify=y_eval_cal_full
                                                                                            )

    
    print(f'Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val_cal.shape}, Test Calibration shape: {X_test_cal.shape}')
    # Convert to PyTorch tensors
    X_train_cal = torch.tensor(X_train_cal, dtype=torch.float32)
    y_train_cal = torch.tensor(y_train_cal, dtype=torch.long)
    p_train_cal = torch.tensor(p_train_cal, dtype=torch.long)

    X_test_cal = torch.tensor(X_test_cal, dtype=torch.float32)
    y_test_cal = torch.tensor(y_test_cal, dtype=torch.long)
    p_test_cal = torch.tensor(p_test_cal, dtype=torch.long)
    
    X_val_cal = torch.tensor(X_val_cal, dtype=torch.float32)
    y_val_cal = torch.tensor(y_val_cal, dtype=torch.long)
    p_val_cal = torch.tensor(p_val_cal, dtype=torch.long)

    # Create datasets
    train_cal_set = CalibrationDataset(X_train_cal, y_train_cal,p_train_cal, num_classes=kwargs.dataset.num_classes)
    test_cal_set = CalibrationDataset(X_test_cal, y_test_cal, p_test_cal, num_classes=kwargs.dataset.num_classes)
    val_cal_set = CalibrationDataset(X_val_cal, y_val_cal, p_val_cal, num_classes=kwargs.dataset.num_classes)
    
    # Create data loaders
    data_train_cal_loader = DataLoader(
        train_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    data_test_cal_loader = DataLoader(
        test_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    data_val_cal_loader = DataLoader(
        val_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    return data_train_cal_loader, data_test_cal_loader, data_val_cal_loader 

def generateCalibrationDatav2(kwargs, dataname=None):
    #temperature = str(int(kwargs.checkpoint.temperature))
    
    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "fog",
        "snow",
        "frost", # try        
        "brightness", # good
        "contrast",
        "pixelate",        
    ]
    
    if (kwargs.corruption_type) and (kwargs.corruption_type not in corruptions):
        raise ValueError(f'Unknown corruption type! {kwargs.corruption_type} was given.')

    if kwargs.data == 'synthetic':   
        test_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
        'pre-train',
        kwargs.data,
        kwargs.checkpoint.num_classes,
        kwargs.checkpoint.num_features,
        kwargs.checkpoint.seed,
        kwargs.checkpoint.epochs,
        kwargs.checkpoint.temperature       
        )
        cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            'pre-train',
            kwargs.data,
            kwargs.checkpoint.num_classes,
            kwargs.checkpoint.num_features,
            kwargs.checkpoint.seed,
            kwargs.checkpoint.epochs,
            kwargs.checkpoint.temperature
        )           
    else:        
        if kwargs.corruption_type:
            cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                'pre-train',
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                kwargs.corruption_type+f"_severity_{kwargs.severity}", # this data should not be corrupted actually! 
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature       
            )
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                'pre-train',
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                kwargs.corruption_type+f"_severity_{kwargs.severity}",
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature
            )  
            # DATA TO TAKE THE VALIDATION SET FROM, SHOULD NOT BE CORRUPTED
            clean_test_results = "results/{}/{}_{}_classes_{}_features/raw_results_clean_eval_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                'pre-train',
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                kwargs.corruption_type+f"_severity_{kwargs.severity}",
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature
            )  
        else:
            cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                'pre-train',
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature       
            )
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                'pre-train',
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature
            )    
            if kwargs.data == 'weather':
                test_shift_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_shift_seed-{}_ep-{}_tmp_{}.csv".format(
                    'pre-train',
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.checkpoint.seed,
                    kwargs.checkpoint.epochs,
                    kwargs.checkpoint.temperature
                )
        
        # if kwargs.data == 'food101':
        #     val_results = "results/{}/{}_{}_classes_{}_features/raw_results_val_cal_seed-{}_ep-{}_tmp_{}.csv".format(
        #         'pre-train',
        #         kwargs.data,
        #         kwargs.dataset.num_classes,
        #         kwargs.dataset.num_features,
        #         kwargs.checkpoint.seed,
        #         kwargs.checkpoint.epochs,
        #         kwargs.checkpoint.temperature
        #     )    
    
    # Load your data
    df_train_calibration_data = pd.read_csv(cal_results)
    if kwargs.data == 'weather' and kwargs.dataset.shift:
        df_eval_calibration_data = pd.read_csv(test_shift_results)        
        df_in_test = pd.read_csv(test_results)
    else:
        df_eval_calibration_data = pd.read_csv(test_results)
        if kwargs.corruption_type:
            df_in_test = pd.read_csv(clean_test_results)
    # if kwargs.data == 'food101':
    #     df_val_calibration_data = pd.read_csv(val_results)    

    # Extract features and labels
    cols = df_train_calibration_data.columns
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logits")]
    pca_cols      = [c for c in cols if c.startswith("pca")]
    # Extract values
    feats_train_cal  = df_train_calibration_data[features_cols].values
    logits_train_cal = df_train_calibration_data[logits_cols].values
    pca_train_cal    = df_train_calibration_data[pca_cols].values
    #feats_train_cal = df_train_calibration_data.filter(regex=r'^features').values #df_train_calibration_data.drop(columns=["true", "preds"]).values
    #logits_train_cal = df_train_calibration_data.filter(regex=r'^logits').values
    #pca_train_cal = df_train_calibration_data.filter(regex=r'^pca').values
    y_train_cal = df_train_calibration_data["true"].values
    p_train_cal = df_train_calibration_data["preds"].values
    # std_per_column = np.std(feats_train_cal, axis=0)
    # print("Standard deviation per column:")
    # print(std_per_column)
    # print("Mean standard deviation per column:")
    # print(np.mean(std_per_column))
    
    cols = df_eval_calibration_data.columns
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logits")]
    pca_cols      = [c for c in cols if c.startswith("pca")]
    # Extract values
    feats_eval_cal  = df_eval_calibration_data[features_cols].values
    logits_eval_cal = df_eval_calibration_data[logits_cols].values
    pca_eval_cal    = df_eval_calibration_data[pca_cols].values
    #feats_eval_cal = df_eval_calibration_data.filter(regex=r'^features').values #df_train_calibration_data.drop(columns=["true", "preds"]).values
    #logits_eval_cal = df_eval_calibration_data.filter(regex=r'^logits').values
    #pca_eval_cal = df_eval_calibration_data.filter(regex=r'^pca').values
    y_eval_cal = df_eval_calibration_data["true"].values
    p_eval_cal = df_eval_calibration_data["preds"].values    
    
    if (kwargs.data == 'weather' and kwargs.dataset.shift) or kwargs.corruption_type:
        cols = df_in_test.columns
        # Single pass grouping
        features_cols = [c for c in cols if c.startswith("features")]
        logits_cols   = [c for c in cols if c.startswith("logits")]
        pca_cols      = [c for c in cols if c.startswith("pca")]
        # Extract values
        feats_in_test  = df_in_test[features_cols].values
        logits_in_test = df_in_test[logits_cols].values
        pca_in_test    = df_in_test[pca_cols].values
        #feats_eval_cal_full = df_in_test.filter(regex=r'^features').values #df_train_calibration_data.drop(columns=["true", "preds"]).values
        #logits_eval_cal_full = df_in_test.filter(regex=r'^logits').values
        #pca_eval_cal_full = df_in_test.filter(regex=r'^pca').values
        y_in_test = df_in_test["true"].values
        p_in_test = df_in_test["preds"].values
    
    # if kwargs.data == 'food101':
    #     cols = df_val_calibration_data.columns
    #     # Single pass grouping
    #     features_cols = [c for c in cols if c.startswith("features")]
    #     logits_cols   = [c for c in cols if c.startswith("logits")]
    #     pca_cols      = [c for c in cols if c.startswith("pca")]
    #     # Extract values
    #     feats_val_cal  = df_val_calibration_data[features_cols].values
    #     logits_val_cal = df_val_calibration_data[logits_cols].values
    #     pca_val_cal    = df_val_calibration_data[pca_cols].values
    #     #feats_eval_cal = df_eval_calibration_data.filter(regex=r'^features').values #df_train_calibration_data.drop(columns=["true", "preds"]).values
    #     #logits_eval_cal = df_eval_calibration_data.filter(regex=r'^logits').values
    #     #pca_eval_cal = df_eval_calibration_data.filter(regex=r'^pca').values
    #     y_val_cal = df_val_calibration_data["true"].values
    #     p_val_cal = df_val_calibration_data["preds"].values
    # else:
    
    # Split into 90% test and 10% val
    #if kwargs.data == 'weather' or kwargs.data == 'tissue' or kwargs.corruption_type:                 
    if (kwargs.data == 'weather' and kwargs.dataset.shift) or kwargs.corruption_type:
        (feats_in_test, feats_val,
        logits_in_test, logits_val,
        pca_in_test, pca_val,
        y_in_test, y_val,
        p_in_test, p_val) = train_test_split(
            feats_in_test,
            logits_in_test,
            pca_in_test,
            y_in_test,
            p_in_test,
            test_size=0.1, #0.1  # 10% for validation
            random_state=kwargs.seed, # for reproducibility
            shuffle=True)    
        
        feats_test = feats_eval_cal
        logits_test = logits_eval_cal
        pca_test = pca_eval_cal
        y_test = y_eval_cal
        p_test = p_eval_cal     
    else:
        (feats_test, feats_val,
        logits_test, logits_val,
        pca_test, pca_val,
        y_test, y_val,
        p_test, p_val) = train_test_split(
            feats_eval_cal,
            logits_eval_cal,
            pca_eval_cal,
            y_eval_cal,
            p_eval_cal,
            test_size=0.1, #0.1  # 10% for validation
            random_state=kwargs.seed, # for reproducibility
            shuffle=True)  
        
    if kwargs.data == 'weather' or kwargs.data == 'tissue':    
        if kwargs.dataset.subsample < 1.0:
            discard_size = 1 - kwargs.dataset.subsample
            
            (feats_train_cal, _,
            logits_train_cal, _,
            pca_train_cal, _,
            y_train_cal, _,
            p_train_cal, _) = train_test_split(
                feats_train_cal,
                logits_train_cal,
                pca_train_cal,
                y_train_cal,
                p_train_cal,
                test_size=discard_size, #0.1  # 10% for validation
                random_state=kwargs.seed, # for reproducibility
                shuffle=True)                        
                                            
            (feats_val, _,
            logits_val, _,
            pca_val, _,
            y_val, _,
            p_val, _) = train_test_split(
                feats_val,
                logits_val,
                pca_val,
                y_val,
                p_val,
                test_size=discard_size, #0.1  # 10% for validation
                random_state=kwargs.seed, # for reproducibility
                shuffle=True)                              
    # else:
    #     (feats_test, feats_val,
    #     logits_test, logits_val,
    #     pca_test, pca_val,
    #     y_test, y_val,
    #     p_test, p_val) = train_test_split(
    #         feats_eval_cal,
    #         logits_eval_cal,
    #         pca_eval_cal,
    #         y_eval_cal,
    #         p_eval_cal,
    #         test_size=0.1, #0.1  # 10% for validation
    #         random_state=kwargs.seed, # for reproducibility
    #         shuffle=True)        
    
    print(f'Learn Calibration shape: {feats_train_cal.shape}, Validation shape: {feats_val.shape}, Test Calibration shape: {feats_test.shape}')
    # Convert to PyTorch tensors
    feats_train_cal = torch.tensor(feats_train_cal, dtype=torch.float32)
    logits_train_cal = torch.tensor(logits_train_cal, dtype=torch.float32)
    pca_train_cal = torch.tensor(pca_train_cal, dtype=torch.float32)
    y_train_cal = torch.tensor(y_train_cal, dtype=torch.long)
    p_train_cal = torch.tensor(p_train_cal, dtype=torch.long)

    feats_test_cal = torch.tensor(feats_test, dtype=torch.float32)
    logits_test_cal = torch.tensor(logits_test, dtype=torch.float32)
    pca_test_cal = torch.tensor(pca_test, dtype=torch.float32)
    y_test_cal = torch.tensor(y_test, dtype=torch.long)
    p_test_cal = torch.tensor(p_test, dtype=torch.long)
    
    feats_val_cal = torch.tensor(feats_val, dtype=torch.float32)
    logits_val_cal = torch.tensor(logits_val, dtype=torch.float32)
    pca_val_cal = torch.tensor(pca_val, dtype=torch.float32)
    y_val_cal = torch.tensor(y_val, dtype=torch.long)
    p_val_cal = torch.tensor(p_val, dtype=torch.long)

    # Create datasets
    train_cal_set = CalibrationDatasetv2(feats_train_cal, logits_train_cal, pca_train_cal, y_train_cal,p_train_cal, num_classes=kwargs.dataset.num_classes)
    test_cal_set = CalibrationDatasetv2(feats_test_cal, logits_test_cal, pca_test_cal, y_test_cal, p_test_cal, num_classes=kwargs.dataset.num_classes)
    val_cal_set = CalibrationDatasetv2(feats_val_cal, logits_val_cal, pca_val_cal, y_val_cal, p_val_cal, num_classes=kwargs.dataset.num_classes)
    
    # Create data loaders
    data_train_cal_loader = DataLoader(
        train_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    data_test_cal_loader = DataLoader(
        test_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    data_val_cal_loader = DataLoader(
        val_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    return data_train_cal_loader, data_test_cal_loader, data_val_cal_loader 


def generatefoodCalibrationData(kwargs, dataname=None):
    #temperature = str(int(kwargs.checkpoint.temperature))
    epochs = 'None'
    temperature = 1.0
            
    cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
        'pre-train',
        kwargs.data,
        kwargs.dataset.num_classes,
        kwargs.dataset.num_features,
        kwargs.checkpoint.seed,
        epochs,
        temperature       
    )
    test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
        'pre-train',
        kwargs.data,
        kwargs.dataset.num_classes,
        kwargs.dataset.num_features,
        kwargs.checkpoint.seed,
        epochs,
        temperature
    )    
    val_results = "results/{}/{}_{}_classes_{}_features/raw_results_val_cal_seed-{}_ep-{}_tmp_{}.csv".format(
        'pre-train',
        kwargs.data,
        kwargs.dataset.num_classes,
        kwargs.dataset.num_features,
        kwargs.checkpoint.seed,
        epochs,
        temperature
    )    
        
    # Load your data
    df_train_calibration_data = pd.read_csv(cal_results)
    df_eval_calibration_data = pd.read_csv(test_results)    
    df_val_data = pd.read_csv(val_results)

    # Extract features and labels
    cols = df_train_calibration_data.columns
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logit")]
    pca_cols      = [c for c in cols if c.startswith("pca")]
    
    # Extract values
    feats_train_cal  = df_train_calibration_data[features_cols].values
    logits_train_cal = df_train_calibration_data[logits_cols].values
    pca_train_cal    = df_train_calibration_data[pca_cols].values

    y_train_cal = df_train_calibration_data["true"].values
    p_train_cal = df_train_calibration_data["preds"].values
    
    cols = df_eval_calibration_data.columns
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logit")]
    pca_cols      = [c for c in cols if c.startswith("pca")]
    # Extract values
    feats_test  = df_eval_calibration_data[features_cols].values
    logits_test = df_eval_calibration_data[logits_cols].values
    pca_test    = df_eval_calibration_data[pca_cols].values

    y_test = df_eval_calibration_data["true"].values
    p_test = df_eval_calibration_data["preds"].values
    
    cols = df_val_data.columns
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logit")]
    pca_cols      = [c for c in cols if c.startswith("pca")]
    # Extract values
    feats_val  = df_val_data[features_cols].values
    logits_val = df_val_data[logits_cols].values
    pca_val    = df_val_data[pca_cols].values
        
    y_val = df_val_data["true"].values
    p_val = df_val_data["preds"].values   
    
    print(f'Learn Calibration shape: {feats_train_cal.shape}, Validation shape: {feats_val.shape}, Test Calibration shape: {feats_test.shape}')
    # Convert to PyTorch tensors
    feats_train_cal = torch.tensor(feats_train_cal, dtype=torch.float32)
    logits_train_cal = torch.tensor(logits_train_cal, dtype=torch.float32)
    pca_train_cal = torch.tensor(pca_train_cal, dtype=torch.float32)
    y_train_cal = torch.tensor(y_train_cal, dtype=torch.long)
    p_train_cal = torch.tensor(p_train_cal, dtype=torch.long)

    feats_test_cal = torch.tensor(feats_test, dtype=torch.float32)
    logits_test_cal = torch.tensor(logits_test, dtype=torch.float32)
    pca_test_cal = torch.tensor(pca_test, dtype=torch.float32)
    y_test_cal = torch.tensor(y_test, dtype=torch.long)
    p_test_cal = torch.tensor(p_test, dtype=torch.long)
    
    feats_val_cal = torch.tensor(feats_val, dtype=torch.float32)
    logits_val_cal = torch.tensor(logits_val, dtype=torch.float32)
    pca_val_cal = torch.tensor(pca_val, dtype=torch.float32)
    y_val_cal = torch.tensor(y_val, dtype=torch.long)
    p_val_cal = torch.tensor(p_val, dtype=torch.long)

    # Create datasets
    train_cal_set = CalibrationDatasetv2(feats_train_cal, logits_train_cal, pca_train_cal, y_train_cal, p_train_cal, num_classes=kwargs.dataset.num_classes)
    test_cal_set = CalibrationDatasetv2(feats_test_cal, logits_test_cal, pca_test_cal, y_test_cal, p_test_cal, num_classes=kwargs.dataset.num_classes)
    val_cal_set = CalibrationDatasetv2(feats_val_cal, logits_val_cal, pca_val_cal, y_val_cal, p_val_cal, num_classes=kwargs.dataset.num_classes)
    
    # Create data loaders
    data_train_cal_loader = DataLoader(
        train_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    data_test_cal_loader = DataLoader(
        test_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    data_val_cal_loader = DataLoader(
        val_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    return data_train_cal_loader, data_test_cal_loader, data_val_cal_loader 


def generatefoodDataforPca(kwargs):
    #temperature = str(int(kwargs.checkpoint.temperature))
            
    test_results = "results/{}/{}_{}_classes_{}_features/test_set_food101_seed{}.pkl".format(
        'pre-train',
        kwargs.data,
        kwargs.dataset.num_classes,
        kwargs.dataset.num_features,
        kwargs.checkpoint.seed                
    )
    cal_results = "results/{}/{}_{}_classes_{}_features/cal_set_food101_seed{}.pkl".format(
        'pre-train',
        kwargs.data,
        kwargs.dataset.num_classes,
        kwargs.dataset.num_features,
        kwargs.checkpoint.seed
    )        
    val_results = "results/{}/{}_{}_classes_{}_features/hold_set_food101_seed{}.pkl".format(
        'pre-train',
        kwargs.data,
        kwargs.dataset.num_classes,
        kwargs.dataset.num_features,
        kwargs.checkpoint.seed
    )    
        
    # Load your data
    df_eval_calibration_data = pd.read_pickle(test_results)
    df_train_calibration_data = pd.read_pickle(cal_results)        
    df_val_data = pd.read_pickle(val_results)

    # Extract features and labels
    cols = df_train_calibration_data.columns
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logit")]    
    
    # Extract values
    feats_train_cal  = df_train_calibration_data[features_cols].values
    logits_train_cal = df_train_calibration_data[logits_cols].values    

    y_train_cal = df_train_calibration_data["true_labels"].values
    p_train_cal = df_train_calibration_data["predicted_labels"].values
    
    # Extract features and labels
    cols = df_eval_calibration_data.columns    
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logit")]
    # Extract values
    feats_test  = df_eval_calibration_data[features_cols].values
    logits_test = df_eval_calibration_data[logits_cols].values    

    y_test = df_eval_calibration_data["true_labels"].values
    p_test = df_eval_calibration_data["predicted_labels"].values
    
    # Extract features and labels
    cols = df_val_data.columns  
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logit")]  
    # Extract values
    feats_val  = df_val_data[features_cols].values
    logits_val = df_val_data[logits_cols].values    
        
    y_val = df_val_data["true_labels"].values
    p_val = df_val_data["predicted_labels"].values   
    
    print(f'Learn Calibration shape: {feats_train_cal.shape}, Validation shape: {feats_val.shape}, Test Calibration shape: {feats_test.shape}')
    # Convert to PyTorch tensors
    feats_train_cal = torch.tensor(feats_train_cal, dtype=torch.float32)
    logits_train_cal = torch.tensor(logits_train_cal, dtype=torch.float32)    
    y_train_cal = torch.tensor(y_train_cal, dtype=torch.long)
    p_train_cal = torch.tensor(p_train_cal, dtype=torch.long)

    feats_test_cal = torch.tensor(feats_test, dtype=torch.float32)
    logits_test_cal = torch.tensor(logits_test, dtype=torch.float32)    
    y_test_cal = torch.tensor(y_test, dtype=torch.long)
    p_test_cal = torch.tensor(p_test, dtype=torch.long)
    
    feats_val_cal = torch.tensor(feats_val, dtype=torch.float32)
    logits_val_cal = torch.tensor(logits_val, dtype=torch.float32)    
    y_val_cal = torch.tensor(y_val, dtype=torch.long)
    p_val_cal = torch.tensor(p_val, dtype=torch.long)

    # Create datasets
    train_cal_set = FoodCalibrationDataset(feats_train_cal, logits_train_cal, y_train_cal, p_train_cal, num_classes=kwargs.dataset.num_classes)
    test_cal_set = FoodCalibrationDataset(feats_test_cal, logits_test_cal, y_test_cal, p_test_cal, num_classes=kwargs.dataset.num_classes)
    val_cal_set = FoodCalibrationDataset(feats_val_cal, logits_val_cal, y_val_cal, p_val_cal, num_classes=kwargs.dataset.num_classes)
    
    # Create data loaders
    data_train_cal_loader = DataLoader(
        train_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    data_test_cal_loader = DataLoader(
        test_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    data_val_cal_loader = DataLoader(
        val_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    return data_train_cal_loader, data_test_cal_loader, data_val_cal_loader


class CalibrationDataset(Dataset):
    def __init__(self, X, y, p, num_classes, transforms_fn=None):
        self.num_classes = num_classes
        self.transforms_fn = transforms_fn
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded
        self.p = torch.tensor(p, dtype=torch.long)  # one-hot encoded
        self.y_onehot = F.one_hot(self.y, num_classes=self.num_classes).float()
        self.p_onehot = F.one_hot(self.p, num_classes=self.num_classes).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transforms_fn is not None:
            x = self.transforms_fn(self.X[idx])
        else:
            x = self.X[idx]
        y_onehot = self.y_onehot[idx]
        p_onehot = self.p_onehot[idx]
        p = self.p[idx]
        return x, y_onehot, p, p_onehot
    
class CalibrationDatasetv2(Dataset):
    def __init__(self, feats, logits, pca, y, p, num_classes, transforms_fn=None):
        self.num_classes = num_classes
        self.transforms_fn = transforms_fn
        self.feats = torch.tensor(feats, dtype=torch.float32)
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.pca = torch.tensor(pca, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded
        self.p = torch.tensor(p, dtype=torch.long)  # one-hot encoded
        self.y_onehot = F.one_hot(self.y, num_classes=self.num_classes).float()
        self.p_onehot = F.one_hot(self.p, num_classes=self.num_classes).float()        

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        if self.transforms_fn is not None:
            feats = self.transforms_fn(self.feats[idx])
        else:
            feats = self.feats[idx]        
        logits = self.logits[idx]
        pca = self.pca[idx]
        y_onehot = self.y_onehot[idx]
        p_onehot = self.p_onehot[idx]
        p = self.p[idx]
        return feats, logits, pca, y_onehot, p, p_onehot


class ClassificationDataset(Dataset):
    def __init__(self, X, y, transforms_fn=None, as_array=False):
        if as_array:
            self.X = X #torch.tensor(X, dtype=torch.float32)        
        else:
            self.X = torch.tensor(X, dtype=torch.float32)        
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded
        self.transforms_fn = transforms_fn
        self.as_array = as_array

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]   
        if self.as_array:                         
            x = Image.fromarray((x * 255).astype(np.uint8)) if x.max() <= 1.0 else Image.fromarray(x.astype(np.uint8))

        if self.transforms_fn is not None:
            x = self.transforms_fn(x)
            
        y = self.y[idx]
        return x, y

class WeatherClassificationDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)  
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)         
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded                

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_num = self.X_num[idx]           
        x_cat = self.X_cat[idx]           
        y = self.y[idx]
        return (x_cat, x_num), y 
        
class OttoClassificationDataset(Dataset):
    def __init__(self, X, y):        
        self.X = torch.tensor(X, dtype=torch.float32)       
        self.X_cat = torch.empty((self.X.shape[0], 0), dtype=torch.long) 
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_num = self.X[idx]           
        x_cat = self.X_cat[idx]           
        y = self.y[idx]
        return (x_cat, x_num), y 
        
class CovTypeClassificationDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)  
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)         
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded                

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_num = self.X_num[idx]           
        x_cat = self.X_cat[idx]           
        y = self.y[idx]
        return (x_cat, x_num), y    


class SynthData(Dataset):
    def __init__(self, kwargs, experiment=None, name='synthetic'):        
        self.dataname = name
        if experiment == 'pre-train':        
            self.generatePretrainingSynthData(num_features = kwargs.num_features,
                                    num_classes = kwargs.num_classes,
                                    n_samples = kwargs.num_samples,
                                    n_redundant = kwargs.n_redundant,
                                    n_clusters_per_class = kwargs.n_clusters_per_class,
                                    class_sep = kwargs.class_sep,
                                    flip_y = kwargs.flip_y,
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition':
            self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs, dataname=self.dataname) 
            print("Loading synthetic data for calibration complete")   

    def generatePretrainingSynthData(self, num_features,
                                num_classes,
                                n_samples,
                                n_redundant,
                                n_clusters_per_class,
                                class_sep,
                                flip_y,
                                batch_size,
                                random_state):        
        
        X, y = make_classification(
        n_samples=n_samples, #240000
        n_features=num_features,        # 2D input
        n_informative=num_features,     # both features informative
        n_redundant=n_redundant, #0
        n_clusters_per_class=n_clusters_per_class, #1
        n_classes=num_classes,
        class_sep=class_sep, #0.5
        flip_y=flip_y, #0.1
        random_state=random_state #42
    )

        X = StandardScaler().fit_transform(X)  # normalize input

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_state)
        X_train, X_train_cal, y_train, y_train_cal = train_test_split(X, y, test_size=0.5, random_state=random_state)
        #X_cal, X_test, y_cal, y_test = train_test_split(X_test, y_test, test_size=0.1668, random_state=random_state)        
        X_eval_cal, X_train_cal, y_eval_cal, y_train_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.1668, random_state=random_state)        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1668, random_state=random_state)
        #y_train_oh = F.one_hot(y_train, num_classes=num_classes) #to_categorical(y_train, num_classes)
      
        print(f'Train shape: {X_train.shape}, Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val.shape}, Eval Calibration shape: {X_eval_cal.shape}')
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long) 
        X_train_cal = torch.tensor(X_train_cal, dtype=torch.float32)
        y_train_cal = torch.tensor(y_train_cal, dtype=torch.long) 
        X_eval_cal = torch.tensor(X_eval_cal, dtype=torch.float32)
        y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long) 
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long) 
        
        train_set = ClassificationDataset(X_train, y_train)
        eval_cal_set   = ClassificationDataset(X_eval_cal, y_eval_cal)
        val_set   = ClassificationDataset(X_val, y_val)
        train_cal_set  = ClassificationDataset(X_train_cal, y_train_cal)       

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading synthetic data for pre-training complete")            
            
            
class CovTypeData(Dataset):
    def __init__(self, kwargs, experiment=None, name='covtype'):        
        self.dataname = name
        if experiment == 'pre-train':        
            self.generatePretrainingCovTypeData(batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition':
            self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs, dataname=self.dataname) 
            print("Loading synthetic data for calibration complete")   

    def generatePretrainingCovTypeData(self, 
                                batch_size,
                                random_state):        
        
        forest_cover_type = pd.read_csv('./data/COVERTYPE/covtype.csv') 
        self.categorical_features = ['Wilderness', 'Soil']        
        self.numerical_features = []
        for col in forest_cover_type.columns:
            if col != 'Cover_Type':
                if not ( col.startswith('Wilderness') or col.startswith('Soil_Type') ):                    
                    self.numerical_features.append(col)
                 
        forest_cover_type['Wilderness'] = forest_cover_type[[col for col in forest_cover_type.columns if col.startswith('Wilderness')]].idxmax(axis=1).apply(lambda x: int(''.join(filter(str.isdigit, x)))-1)
        forest_cover_type['Soil'] = forest_cover_type[[col for col in forest_cover_type.columns if col.startswith('Soil_Type')]].idxmax(axis=1).apply(lambda x: int(''.join(filter(str.isdigit, x)))-1)
        forest_cover_type.drop(columns=[col for col in forest_cover_type.columns if col.startswith('Wilderness_Area') or col.startswith('Soil_Type')], inplace=True)        
        self.category_counts = tuple(forest_cover_type[feature].nunique() for feature in self.categorical_features)
        print(forest_cover_type.head(5))
        
        X_num = forest_cover_type.drop(columns=self.categorical_features+['Cover_Type'])
        X_cat = forest_cover_type.drop(columns=self.numerical_features+['Cover_Type'])
        y = forest_cover_type['Cover_Type'].values - 1 # from 0 to 6 not from 1 to 7!!!!

        X_num = StandardScaler().fit_transform(X_num)  # normalize input        
        X_cat_array = X_cat.values
        X = np.concatenate([X_num, X_cat_array], axis=1)

        X_train, X_train_cal, y_train, y_train_cal = train_test_split(X, y, test_size=0.5, random_state=random_state)
        X_eval_cal, X_train_cal, y_eval_cal, y_train_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.16, random_state=random_state)        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.16, random_state=random_state)   
        print(f'Train shape: {X_train.shape}, Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val.shape}, Eval Calibration shape: {X_eval_cal.shape}')
        
        X_train_num = X_train[:, :-2]
        X_train_cat = X_train[:, -2:]
        X_val_num = X_val[:, :-2]
        X_val_cat = X_val[:, -2:]
        X_train_cal_num = X_train_cal[:, :-2]
        X_train_cal_cat = X_train_cal[:, -2:]
        X_eval_cal_num = X_eval_cal[:, :-2]
        X_eval_cal_cat = X_eval_cal[:, -2:]
                  
        X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
        X_train_cat = torch.tensor(X_train_cat, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long) 
        X_train_cal_num = torch.tensor(X_train_cal_num, dtype=torch.float32)
        X_train_cal_cat = torch.tensor(X_train_cal_cat, dtype=torch.float32)
        y_train_cal = torch.tensor(y_train_cal, dtype=torch.long) 
        X_eval_cal_num = torch.tensor(X_eval_cal_num, dtype=torch.float32)
        X_eval_cal_cat = torch.tensor(X_eval_cal_cat, dtype=torch.float32)
        y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long) 
        X_val_num = torch.tensor(X_val_num, dtype=torch.float32)
        X_val_cat = torch.tensor(X_val_cat, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long) 
        
        train_set = CovTypeClassificationDataset(X_train_num, X_train_cat, y_train)
        eval_cal_set   = CovTypeClassificationDataset(X_eval_cal_num, X_eval_cal_cat, y_eval_cal)
        val_set   = CovTypeClassificationDataset(X_val_num, X_val_cat, y_val)
        train_cal_set  = CovTypeClassificationDataset(X_train_cal_num, X_train_cal_cat, y_train_cal)             

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading covtype data for pre-training complete")                      

         
class OttoData(Dataset):
    def __init__(self, kwargs, experiment=None, name='otto'):        
        self.dataname = name
        if experiment == 'pre-train':        
            self.generatePretrainingOttoData(batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition':
            self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs, dataname=self.dataname) 
            print("Loading synthetic data for calibration complete")   

    def generatePretrainingOttoData(self, 
                                batch_size,
                                random_state):        
        
        otto = pd.read_csv('./data/OTTO/train.csv')  # 60k obs                        
        
        X = otto.drop(columns=['id', 'target'])    
        self.numerical_features = X.shape[1]    
        otto['target_int'] = otto['target'].map(lambda x: int(x.split('_')[1]))
        y = otto['target_int'].values - 1 # from 0 to 8 not from 1 to 9!!!!

        X = StandardScaler().fit_transform(X)  # normalize input                        

        X_train, X_train_cal, y_train, y_train_cal = train_test_split(X, y, test_size=0.5, random_state=random_state)
        X_eval_cal, X_train_cal, y_eval_cal, y_train_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.16, random_state=random_state)        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.16, random_state=random_state)   
        print(f'Train shape: {X_train.shape}, Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val.shape}, Eval Calibration shape: {X_eval_cal.shape}')
                
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long) 
        X_train_cal = torch.tensor(X_train_cal, dtype=torch.float32)
        y_train_cal = torch.tensor(y_train_cal, dtype=torch.long) 
        X_eval_cal = torch.tensor(X_eval_cal, dtype=torch.float32)
        y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long) 
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long) 
        
        train_set = OttoClassificationDataset(X_train, y_train)
        eval_cal_set   = OttoClassificationDataset(X_eval_cal, y_eval_cal)
        val_set   = OttoClassificationDataset(X_val, y_val)
        train_cal_set  = OttoClassificationDataset(X_train_cal, y_train_cal)          

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading otto data for pre-training complete")   

class WeatherData(Dataset):
    def __init__(self, kwargs, experiment=None, name='wheather', seed=42):        
        self.dataname = name
        if experiment == 'pre-train':   
            self.generatePretrainingWheatherData(batch_size=kwargs.batch_size, seed=seed, exp_name=experiment)      
        elif experiment == 'xg_debug':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_train_cal, self.y_train_cal, self.X_eval_cal, self.y_eval_cal, self.X_eval_cal_shift, self.y_eval_cal_shift = self.generatePretrainingWheatherData(batch_size=kwargs.batch_size, seed=seed, exp_name=experiment)      
        else:
            if kwargs.calibrator_version == 'v2':
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
            else:
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 

    def generatePretrainingWheatherData(self, seed, batch_size, exp_name='pre-train'):        
        
        scaler = StandardScaler()
        ############  FOR ALL CONTINUOUS ############ 
        self.category_counts = ()        
        
        ######## TRAINING SET ########
        weather = pd.read_csv(f'./data/WEATHER/shifts_canonical_train_seed_{seed}.csv')  # 60k obs        
        train_df, val_df = train_test_split(weather, test_size=0.1, stratify=weather["fact_cwsm_class"], random_state=seed)                
        
        X = train_df.iloc[:,6:] #weather.drop(columns=['id', 'target'])  
        feature_cols = X.columns.tolist()    
        self.numerical_features = feature_cols  
        X_train = scaler.fit_transform(X[feature_cols])                            
        # self.categorical_features, self.numerical_features, _ = find_categorical_columns(X, target_col='fact_cwsm_class',
        #                                                                                            threshold_for_categorical=20)                    
        # X_num_raw = X.drop(columns=self.categorical_features) #+['fact_cwsm_class'])
        # X_cat_raw = X.drop(columns=self.numerical_features) #+['fact_cwsm_class'])
        
        # cat_maps = self.fit_category_maps(X_cat_raw, self.categorical_features)
        label_map = self.fit_label_map(train_df["fact_cwsm_class"])
        
        # self.cat_levels = {}
        # for col in self.categorical_features:
        #     self.cat_levels[col] = pd.Categorical(X_cat[col]).categories
        # for col in X_cat.columns:
        #     try:
        #         X_cat[col] = X_cat[col].astype(int)
        #     except (ValueError, TypeError):
        #         pass
        # X_cat = X_cat.astype('category').apply(lambda col: col.cat.codes)
        # weather['target_int'] = weather['fact_cwsm_class'].astype(int).astype('category').cat.codes
        # y_train = weather['target_int'].values #- 1 # from 0 to 8 not from 1 to 9!!!!                 
        
        # X_num = scaler.fit_transform(X_num_raw)  # normalize input   
        # X_cat, self.category_counts = self.transform_categories(X_cat_raw, self.categorical_features, cat_maps, use_unknown=True)
        y_train = self.transform_labels(train_df["fact_cwsm_class"], label_map)      
        self.class_counts = np.bincount(y_train, minlength=5)   
        # X_train = np.concatenate([X_num, X_cat], axis=1)
        
        ######## VALIDATION SET ########
        X_val_raw = val_df.iloc[:, 6:].copy()
        X_val = scaler.transform(X_val_raw[feature_cols])

        # X_val_num = scaler.transform(X_val_raw[self.numerical_features].copy())
        # X_val_cat, _ = self.transform_categories(
        #     X_val_raw[self.categorical_features].copy(),
        #     self.categorical_features,
        #     cat_maps,
        #     use_unknown=True
        # )
        y_val = self.transform_labels(val_df["fact_cwsm_class"], label_map)

        # X_val = np.concatenate([X_val_num, X_val_cat], axis=1)
        
        ######## TEST SET (IN-DISTRIBUTION) ########        
        eval_in_df = pd.read_csv(f'./data/WEATHER/shifts_canonical_eval_in_seed_{seed}.csv')  # 60k obs   
        
        X = eval_in_df.iloc[:,6:] #weather.drop(columns=['id', 'target'])     
        X_eval_cal = scaler.transform(X[feature_cols])      

        # X_num = scaler.transform(X[self.numerical_features].copy())
        # X_cat, _ = self.transform_categories(
        #     X[self.categorical_features].copy(),
        #     self.categorical_features,
        #     cat_maps,
        #     use_unknown=True
        # )
        y_eval_cal = self.transform_labels(eval_in_df["fact_cwsm_class"], label_map)

        # X_eval_cal = np.concatenate([X_num, X_cat], axis=1)                                     
        
        # X_eval_cal = weather.iloc[:,6:] #weather.drop(columns=['id', 'target'])               
        # X_num = X_eval_cal.drop(columns=self.categorical_features) #+['fact_cwsm_class'])
        # X_cat = X_eval_cal.drop(columns=self.numerical_features) #+['fact_cwsm_class'])
        # for col in X_cat.columns:
        #     try:
        #         X_cat[col] = X_cat[col].astype(int)
        #     except (ValueError, TypeError):
        #         pass
        # X_cat = X_cat.astype('category').apply(lambda col: col.cat.codes)
        # weather['target_int'] = weather['fact_cwsm_class'].astype(int).astype('category').cat.codes
        # y_eval_cal = weather['target_int'].values #- 1 # from 0 to 8 not from 1 to 9!!!!            
        
        # X_eval_cal = scaler.transform(X_num)  # normalize input   
        # X_eval_cal = np.concatenate([X_eval_cal, X_cat.values], axis=1)
        
        ######## TEST SET (SHIFT) ########        
        eval_out_df = pd.read_csv(f'./data/WEATHER/shifts_canonical_eval_out_seed_{seed}.csv')  # 60k obs     
        
        X = eval_out_df.iloc[:,6:] #weather.drop(columns=['id', 'target'])        
        X_eval_cal_shift = scaler.transform(X[feature_cols])      

        # X_num = scaler.transform(X[self.numerical_features].copy())
        # X_cat, _ = self.transform_categories(
        #     X[self.categorical_features].copy(),
        #     self.categorical_features,
        #     cat_maps,
        #     use_unknown=True
        # )
        y_eval_cal_shift = self.transform_labels(eval_out_df["fact_cwsm_class"], label_map)

        # X_eval_cal_shift = np.concatenate([X_num, X_cat], axis=1)      
        
        # X_eval_cal_shift = weather.iloc[:,6:] #weather.drop(columns=['id', 'target'])               
        # X_eval_cal_shift = filter_unseen_rows(X_eval_cal_shift, self.categorical_features, self.cat_levels)
        # weather = weather.loc[X_eval_cal_shift.index].copy()
        # X_num = X_eval_cal_shift[self.numerical_features].copy()
        # X_cat = X_eval_cal_shift[self.categorical_features].copy()
              
        # for col in X_cat.columns:
        #     try:
        #         X_cat[col] = pd.Categorical(X_cat[col], categories=self.cat_levels[col]).codes # X_cat[col] = X_cat[col].astype(int)
        #     except (ValueError, TypeError):
        #         pass
        # X_cat = X_cat.astype('category').apply(lambda col: col.cat.codes)
        # weather['target_int'] = weather['fact_cwsm_class'].astype(int).astype('category').cat.codes
        # y_eval_cal_shift = weather['target_int'].values #- 1 # from 0 to 8 not from 1 to 9!!!!        
        
        # X_eval_cal_shift = scaler.transform(X_num)  # normalize input   
        # X_eval_cal_shift = np.concatenate([X_eval_cal_shift, X_cat.values], axis=1)
        
        ######## CALIBRATION SET ########
        cal_df = pd.read_csv(f'./data/WEATHER/shifts_canonical_dev_in_processed.csv')  # 60k obs                    
        
        X = cal_df.iloc[:,6:] #weather.drop(columns=['id', 'target'])        
        X_train_cal = scaler.transform(X[feature_cols])      

        # X_num = scaler.transform(X[self.numerical_features].copy())
        # X_cat, _ = self.transform_categories(
        #     X[self.categorical_features].copy(),
        #     self.categorical_features,
        #     cat_maps,
        #     use_unknown=True
        # )
        y_train_cal = self.transform_labels(cal_df["fact_cwsm_class"], label_map)

        # X_train_cal = np.concatenate([X_num, X_cat], axis=1)                            
        
        # X_train_cal = weather.iloc[:,6:] #weather.drop(columns=['id', 'target'])                       
        # X_num = X_train_cal.drop(columns=self.categorical_features) #+['fact_cwsm_class'])
        # X_cat = X_train_cal.drop(columns=self.numerical_features) #+['fact_cwsm_class'])              
        # for col in X_cat.columns:
        #     try:
        #         X_cat[col] = X_cat[col].astype(int)
        #     except (ValueError, TypeError):
        #         pass
        # X_cat = X_cat.astype('category').apply(lambda col: col.cat.codes)
        # weather['target_int'] = weather['fact_cwsm_class'].astype(int).astype('category').cat.codes
        # y_train_cal = weather['target_int'].values #- 1 # from 0 to 8 not from 1 to 9!!!!        
        
        # X_train_cal = scaler.transform(X_num)  # normalize input               
        # X_train_cal = np.concatenate([X_train_cal, X_cat.values], axis=1)

        ######## VALIDATION SET (PRE-TRAINING) ########
        # X_train, X_train_cal, y_train, y_train_cal = train_test_split(X, y, test_size=0.5, random_state=random_state)
        # X_eval_cal, X_train_cal, y_eval_cal, y_train_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.16, random_state=random_state)        
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train)   
        print(f'Train shape: {X_train.shape}, Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val.shape}, Eval Calibration shape: {X_eval_cal.shape}')
                
        # X_train = torch.tensor(X_train, dtype=torch.float32)
        # y_train = torch.tensor(y_train, dtype=torch.long) 
        # X_train_cal = torch.tensor(X_train_cal, dtype=torch.float32)
        # y_train_cal = torch.tensor(y_train_cal, dtype=torch.long) 
        # X_eval_cal = torch.tensor(X_eval_cal, dtype=torch.float32)
        # y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long) 
        # X_val = torch.tensor(X_val, dtype=torch.float32)
        # y_val = torch.tensor(y_val, dtype=torch.long) 
        
        # train_set = WeatherClassificationDataset(X_train, y_train)
        # eval_cal_set   = WeatherClassificationDataset(X_eval_cal, y_eval_cal)
        # val_set   = WeatherClassificationDataset(X_val, y_val)
        # train_cal_set  = WeatherClassificationDataset(X_train_cal, y_train_cal)          

        # self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        # self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        # self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        # self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        n_num = len(self.numerical_features)

        X_train_num = X_train[:, :n_num]
        X_train_cat = X_train[:, n_num:]

        X_val_num = X_val[:, :n_num]
        X_val_cat = X_val[:, n_num:]

        X_train_cal_num = X_train_cal[:, :n_num]
        X_train_cal_cat = X_train_cal[:, n_num:]

        X_eval_cal_num = X_eval_cal[:, :n_num]
        X_eval_cal_cat = X_eval_cal[:, n_num:]
        
        X_eval_cal_num_shift = X_eval_cal_shift[:, :n_num]
        X_eval_cal_cat_shift = X_eval_cal_shift[:, n_num:]
        
        # for split_name, X_cat_arr in [
        #     ("train", X_train_cat),
        #     ("val", X_val_cat),
        #     ("train_cal", X_train_cal_cat),
        #     ("eval", X_eval_cal_cat),
        #     ("eval_shift", X_eval_cal_cat_shift),
        # ]:
        #     print(f"\n{split_name}")
        #     for i in range(X_cat_arr.shape[1]):
        #         print(
        #             i,
        #             "min=", X_cat_arr[:, i].min(),
        #             "max=", X_cat_arr[:, i].max(),
        #             "allowed_max=", self.category_counts[i] - 1
        #         )
            
        # X_train_num = X_train[:, :-len(self.numerical_features)]
        # X_train_cat = X_train[:, -len(self.numerical_features):]
        # X_val_num = X_val[:, :-len(self.numerical_features)]
        # X_val_cat = X_val[:, -len(self.numerical_features):]
        # X_train_cal_num = X_train_cal[:, :-len(self.numerical_features)]
        # X_train_cal_cat = X_train_cal[:, -len(self.numerical_features):]
        # X_eval_cal_num = X_eval_cal[:, :-len(self.numerical_features)]
        # X_eval_cal_cat = X_eval_cal[:, -len(self.numerical_features):]
                  
        X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
        X_train_cat = torch.tensor(X_train_cat, dtype=torch.long)
        y_train = torch.tensor(y_train, dtype=torch.long) 
        
        X_train_cal_num = torch.tensor(X_train_cal_num, dtype=torch.float32)
        X_train_cal_cat = torch.tensor(X_train_cal_cat, dtype=torch.long)
        y_train_cal = torch.tensor(y_train_cal, dtype=torch.long) 
        
        X_eval_cal_num = torch.tensor(X_eval_cal_num, dtype=torch.float32)
        X_eval_cal_cat = torch.tensor(X_eval_cal_cat, dtype=torch.long)
        y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long)         
        
        X_eval_cal_num_shift = torch.tensor(X_eval_cal_num_shift, dtype=torch.float32)
        X_eval_cal_cat_shift = torch.tensor(X_eval_cal_cat_shift, dtype=torch.long)
        y_eval_cal_shift = torch.tensor(y_eval_cal_shift, dtype=torch.long) 
        
        X_val_num = torch.tensor(X_val_num, dtype=torch.float32)
        X_val_cat = torch.tensor(X_val_cat, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long) 
        
        train_set = WeatherClassificationDataset(X_train_num, X_train_cat, y_train)
        eval_cal_set   = WeatherClassificationDataset(X_eval_cal_num, X_eval_cal_cat, y_eval_cal)
        eval_cal_set_shift   = WeatherClassificationDataset(X_eval_cal_num_shift, X_eval_cal_cat_shift, y_eval_cal_shift)
        val_set   = WeatherClassificationDataset(X_val_num, X_val_cat, y_val)
        train_cal_set  = WeatherClassificationDataset(X_train_cal_num, X_train_cal_cat, y_train_cal)             

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_eval_cal_shift_loader   = DataLoader(eval_cal_set_shift, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)            
        
        print("Loading weather data for pre-training complete")       
        
        if exp_name == 'xg_debug':
            return X_train, y_train, X_val, y_val, X_train_cal, y_train_cal, X_eval_cal, y_eval_cal, X_eval_cal_shift, y_eval_cal_shift        
        
    def fit_category_maps(self, df, categorical_features):
        cat_maps = {}
        for col in categorical_features:
            cat_maps[col] = pd.Index(pd.Categorical(df[col]).categories)
        return cat_maps

    def transform_categories(self, df, categorical_features, cat_maps, use_unknown=True):
        cols = []
        category_counts = []

        for col in categorical_features:
            categories = cat_maps[col]
            codes = pd.Categorical(df[col], categories=categories).codes

            if use_unknown:
                unknown_idx = len(categories)
                codes = np.where(codes < 0, unknown_idx, codes)
                category_counts.append(len(categories) + 1)
            else:
                if (codes < 0).any():
                    raise ValueError(f"Unseen category in {col}")
                category_counts.append(len(categories))

            cols.append(codes.reshape(-1, 1))

        X_cat = np.concatenate(cols, axis=1) if cols else np.empty((len(df), 0), dtype=np.int64)
        return X_cat, category_counts

    def fit_label_map(self, y):
        return pd.Index(pd.Categorical(y.astype(int)).categories)

    def transform_labels(self, y, label_categories):
        codes = pd.Categorical(y.astype(int), categories=label_categories).codes
        if (codes < 0).any():
            raise ValueError("Unseen label found.")
        return codes       
                    
class MnistData(Dataset):    
    def __init__(self, kwargs, experiment=None, name='mnist'):          
        if experiment == 'pre-train':   
            if kwargs.variant:                  
                self.generatePretrainingNMnistData(variant = kwargs.variant,
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)        
            else:
                self.generatePretrainingMnistData(
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition':
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 
        print("Loading synthetic data for calibration complete")   

    def generatePretrainingMnistData(self, 
                                batch_size,
                                random_state):                        
        
        transform = transforms.ToTensor()
        mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

        # Convert to NumPy arrays
        X = mnist_dataset.data.numpy().astype(np.float32) / 255.0  # Normalize to [0,1]
        X = np.expand_dims(X, axis=1)  # Add channel dimension: (N, 1, 28, 28)
        y = np.array(mnist_dataset.targets)
        
        X_train, X_train_cal, y_train, y_train_cal = train_test_split(X, y, test_size=0.5, random_state=random_state)        
        X_eval_cal, X_train_cal, y_eval_cal, y_train_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.1668, random_state=random_state)        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1668, random_state=random_state)
        
        print(f'Train shape: {X_train.shape}, Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val.shape}, Eval Calibration shape: {X_eval_cal.shape}')
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long) 
        X_train_cal = torch.tensor(X_train_cal, dtype=torch.float32)
        y_train_cal = torch.tensor(y_train_cal, dtype=torch.long) 
        X_eval_cal = torch.tensor(X_eval_cal, dtype=torch.float32)
        y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long) 
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long) 
        
        train_set = ClassificationDataset(X_train, y_train)
        eval_cal_set   = ClassificationDataset(X_eval_cal, y_eval_cal)
        val_set   = ClassificationDataset(X_val, y_val)
        train_cal_set  = ClassificationDataset(X_train_cal, y_train_cal)

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading MNIS data for pre-training complete") 
        
    def generatePretrainingNMnistData(self, variant,
                                batch_size,
                                random_state):                        
        variant_map = {
                'awgn': 'mnist-with-awgn.mat',
                'motion_blur': 'mnist-with-motion-blur.mat',
                'reduced_contrast': 'mnist-with-reduced-contrast-and-awgn.mat'
        }
        mat_path = f"./data/NMNIST/{variant_map[variant]}"
        data = loadmat(mat_path)

        # Load and normalize images
        X = data['train_x'].astype(np.float32) / 255.0  # shape: (60000, 784)
        X = X.reshape(-1, 1, 28, 28)  # shape: (60000, 1, 28, 28)
        X_train_cal = data['test_x'].astype(np.float32) / 255.0  # shape: (60000, 784)
        X_train_cal = X_train_cal.reshape(-1, 1, 28, 28)  # shape: (60000, 1, 28, 28)

        # Decode one-hot labels
        y = np.argmax(data['train_y'], axis=1)  # shape: (60000,)
        y_train_cal = np.argmax(data['test_y'], axis=1)  # shape: (60000,)

        # Split into training, calibration, validation, and eval sets
        X_train, X_eval_cal, y_train, y_eval_cal = train_test_split(X, y, test_size=0.829, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_train_cal = torch.tensor(X_train_cal, dtype=torch.float32)
        y_train_cal = torch.tensor(y_train_cal, dtype=torch.long)
        X_eval_cal = torch.tensor(X_eval_cal, dtype=torch.float32)
        y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        # Wrap in datasets
        train_set = ClassificationDataset(X_train, y_train)
        eval_cal_set = ClassificationDataset(X_eval_cal, y_eval_cal)
        val_set = ClassificationDataset(X_val, y_val)
        train_cal_set = ClassificationDataset(X_train_cal, y_train_cal)

        # Create DataLoaders
        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.data_eval_cal_loader = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print(f"Train shape: {X_train.shape}, Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val.shape}, Eval Calibration shape: {X_eval_cal.shape}")
        print(f"Loading n-MNIST variant '{variant}' for pre-training completed.")
                        
def print_class_frequencies(dataset):  
    class_freqs = []  
    labels = np.array(dataset.labels).squeeze()   # dataset.labels is usually (N,1)
    counts = Counter(labels)
    total = len(labels)
    print("Class frequencies:")
    for cls, count in sorted(counts.items()):
        print(f"  Class {cls}: {count} ({count/total:.2%})")
        class_freqs.append(count/total)
    return class_freqs

class OOData(Dataset):
    def __init__(self, cal_loader, val_loader, test_loader):
        self.data_train_cal_loader = cal_loader
        self.data_val_cal_loader = val_loader
        self.data_test_cal_loader = test_loader

class MedMnistData(Dataset):    
    def __init__(self, kwargs, experiment=None, name='path', model_class=None):
        self.name = name
        self.model_class = model_class
        if experiment == 'pre-train':                     
            self.generatePretrainingMedMnistData(size=kwargs.size,
                                batch_size = kwargs.batch_size,
                                random_state = kwargs.random_state)    
            kwargs.class_freqs = self.class_freqs  
        elif experiment == 'calibrate' or experiment == 'competition' or experiment == 'quantize'  or experiment == 'replicate':
                # kwargs.dataset.class_freqs = [0.321, 0.047, 0.035, 0.093, 0.071, 0.047, 0.237, 0.149]
                if kwargs.calibrator_version == 'v2':
                    self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
                else:
                    self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 
        print("Loading synthetic data for calibration complete")   
        
    def generatePretrainingMedMnistData(self, size,
                                batch_size,
                                random_state):
        if self.model_class == 'convnext':
            # Use ImageNet normalization for ConvNeXt, as it was pre-trained on ImageNet
            l_transform = [ #transforms.Compose(
                transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
            ]
        else:
            l_transform = [transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]
        #if self.name == 'tissue':
        #    l_transforms.append()  # Convert to 3-channel RGB)
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(10),
            #transforms.Resize((224, 224)) if size < 224 else transforms.Lambda(lambda x: x),            
            #transforms.ToTensor(), 
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                    std=[0.229, 0.224, 0.225])
        #]

        # Remove any None or identity transforms
        #l_transforms = [t for t in l_transforms if not isinstance(t, transforms.Lambda)]
        
        # l_transform = transforms.Compose([ 
        #         transforms.Grayscale(num_output_channels=3),                            
        #         transforms.Resize((224, 224)),
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        #         ),
        # ])

        trans = transforms.Compose(l_transform)


        print(f"Dataset source information : MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
        info = INFO['tissuemnist']
        print(info['description'])
        os.makedirs(f"./data/{self.name.upper()}/data_train", exist_ok=True)
        os.makedirs(f"./data/{self.name.upper()}/data_val", exist_ok=True)
        os.makedirs(f"./data/{self.name.upper()}/data_test", exist_ok=True)
        generator = torch.Generator().manual_seed(random_state)
        
        if self.name == 'tissue':
            train_set = TissueMNIST(root="./data/TISSUE/data_train", split="train", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #165k        
            self.class_freqs = print_class_frequencies(train_set)
            train_size = len(train_set)
            half_size = train_size // 2                    
            train_set, eval_cal_set = random_split(train_set, [half_size, train_size - half_size], generator=generator) #82.5k BOTH  
            val_set = TissueMNIST(root="./data/TISSUE/data_val", split="val", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #20k
            train_cal_set = TissueMNIST(root="./data/TISSUE/data_test", split="test", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #40k
        elif self.name == 'path':
            train_set = PathMNIST(root="./data/PATH/data_train", split="train", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #89kk        
            train_size = len(train_set)
            half_size = train_size // 2                    
            train_set, eval_cal_set = random_split(train_set, [half_size, train_size - half_size], generator=generator) #45k BOTH  
            val_set = PathMNIST(root="./data/PATH/data_test", split="test", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #7k
            train_cal_set = PathMNIST(root="./data/PATH/data_val", split="val", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #10k
                
        print(f'Train shape: {len(train_set)}, Learn Calibration shape: {len(train_cal_set)}, Validation shape: {len(val_set)}, Eval Calibration shape: {len(eval_cal_set)}')

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print(f"Loading {self.name} mnist data for pre-training complete.")         


class Cifar10Data(Dataset):    
    def __init__(self, kwargs, experiment=None, name='cifar10', model_class=None):
        self.name = name
        self.model_class = model_class
        if experiment == 'pre-train':  
            kwargs.class_freqs = [1/kwargs.num_classes]*kwargs.num_classes                        
            self.generatePretrainingCifar10Data(
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition' or experiment == 'quantize' or experiment == 'replicate':
            kwargs.dataset.class_freqs = [1/kwargs.dataset.num_classes]*kwargs.dataset.num_classes
            if kwargs.calibrator_version == 'v2':
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
            else:
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 
        #print("Loading synthetic data for calibration complete")   

    def generatePretrainingCifar10Data(self, 
                                batch_size,
                                random_state):  
        data_dir =  f"./data/{self.name.upper()}"                     
        os.makedirs(data_dir, exist_ok=True)
        generator = torch.Generator().manual_seed(random_state)
        if self.model_class == 'convnext':
            # Use ImageNet normalization for ConvNeXt, as it was pre-trained on ImageNet
            l_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
            ])
        else:
            l_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        
        # l_transform = transforms.Compose([ 
        #         transforms.Resize((224, 224)),
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        #         ),
        # ])
        full_train = datasets.CIFAR10(root=data_dir, train=True, 
                                        transform=l_transform, download=True)
        
        train_size = int(0.45 * len(full_train))
        val_size = int(0.1 * len(full_train))
        eval_cal_size = len(full_train) - train_size - val_size
        train_set, val_set, eval_cal_set = random_split(full_train, [train_size, val_size, eval_cal_size], generator=generator)

        train_cal_set = datasets.CIFAR10(root=data_dir, train=False, 
                                        transform=l_transform, download=True)
        
        print(f'Train shape: {len(train_set)}, Learn Calibration shape: {len(train_cal_set)}, Validation shape: {len(val_set)}, Eval Calibration shape: {len(eval_cal_set)}')

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading CIFAR10 data for pre-training complete") 


class Food101Data(Dataset):    
    def __init__(self, kwargs, experiment=None, name='food101'):     
        self.name = name  
        kwargs.dataset.class_freqs = [1/kwargs.dataset.num_classes]*kwargs.dataset.num_classes                 
        if experiment == 'pre-train':                                    
            self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generatefoodDataforPca(kwargs)            
        else:                                        
            # if kwargs.calibrator_version == 'v2':
            #     self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
            # else:
            self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generatefoodCalibrationData(kwargs) 
        #print("Loading synthetic data for calibration complete") 
        

class FoodCalibrationDataset(Dataset):
    def __init__(self, feats, logits, y, p, num_classes, transforms_fn=None):
        self.num_classes = num_classes
        self.transforms_fn = transforms_fn
        self.feats = torch.tensor(feats, dtype=torch.float32)
        self.logits = torch.tensor(logits, dtype=torch.float32)        
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded
        self.p = torch.tensor(p, dtype=torch.long)  # one-hot encoded
        self.y_onehot = F.one_hot(self.y, num_classes=self.num_classes).float()
        self.p_onehot = F.one_hot(self.p, num_classes=self.num_classes).float()        

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        if self.transforms_fn is not None:
            feats = self.transforms_fn(self.feats[idx])
        else:
            feats = self.feats[idx]        
        logits = self.logits[idx]        
        y_onehot = self.y_onehot[idx]
        p_onehot = self.p_onehot[idx]
        p = self.p[idx]
        return feats, logits, y_onehot, p, p_onehot


class Cifar10LongTailData(Dataset):    
    def __init__(self, kwargs, experiment=None, name='cifar10LT'):       
        self.name = name   
        if experiment == 'pre-train':                          
            self.generatePretrainingCifar10Data(
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition':
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 
        print("Loading synthetic data for calibration complete")   

    def generatePretrainingCifar10Data(self, 
                                batch_size,
                                random_state):  
        
        data_dir =  f"./data/{self.name.upper()}"      
        full_train = pd.read_parquet(join(data_dir, "train.parquet"))
        
        # Convert images to numpy arrays
        images = []
        labels = []

        for _, row in full_train.iterrows():
            arr = Image.open(io.BytesIO(row["img"]["bytes"]))  # decode from bytes
            arr = np.array(arr)
            # Normalize only if it's uint8
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32)            
            images.append(arr)
            labels.append(row["label"])

        X = np.stack(images)  # shape: (N, 32, 32, 3)
        y = np.array(labels)  # shape: (N,)    

        X_train, X_train_cal, y_train, y_train_cal = train_test_split(X, y, test_size=0.5, random_state=random_state, stratify=y)
        #X_eval_cal, X_train_cal, y_eval_cal, y_train_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.1668, random_state=random_state, stratify=y_train_cal)        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state, stratify=y_train)        
        
        full_train_cal = pd.read_parquet(join(data_dir, "test.parquet"))
        
        # Convert images to numpy arrays
        images = []
        labels = []

        for _, row in full_train_cal.iterrows():
            arr = Image.open(io.BytesIO(row["img"]["bytes"]))  # decode from bytes
            arr = np.array(arr)
            # Normalize only if it's uint8
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32)            
            images.append(arr)
            labels.append(row["label"])

        X_eval_cal = np.stack(images)  # shape: (N, 32, 32, 3)
        y_eval_cal = np.array(labels)  # shape: (N,)    

        print(f'Train shape: {X_train.shape}, Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val.shape}, Eval Calibration shape: {X_eval_cal.shape}')
        #X_train = torch.tensor(X_train, dtype=torch.float32)
        #y_train = torch.tensor(y_train, dtype=torch.long) 
        #X_train_cal = torch.tensor(X_train_cal, dtype=torch.float32)
        #y_train_cal = torch.tensor(y_train_cal, dtype=torch.long) 
        #X_eval_cal = torch.tensor(X_eval_cal, dtype=torch.float32)
        #y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long) 
        #X_val = torch.tensor(X_val, dtype=torch.float32)
        #y_val = torch.tensor(y_val, dtype=torch.long) 
        
        print_class_distribution("Train", y_train)
        print_class_distribution("Validation", y_val)
        print_class_distribution("Learn Calibration", y_train_cal)
        print_class_distribution("Eval Calibration", y_eval_cal)
        
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),               # match ViT input size
            transforms.RandomHorizontalFlip(p=0.5),      # common for CIFAR
            transforms.RandomCrop(224, padding=4),       # adds spatial jitter
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # brightness, contrast, saturation, hue
            transforms.RandomRotation(15),               # small rotations
            transforms.RandomGrayscale(p=0.1),           # sometimes grayscale
            transforms.ToTensor(),
            transforms.Normalize(                        # use ImageNet mean/std for ViT
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        train_set = ClassificationDataset(X_train, y_train, transforms_fn=train_transforms)
        eval_cal_set   = ClassificationDataset(X_eval_cal, y_eval_cal, transforms_fn=val_transforms)
        val_set   = ClassificationDataset(X_val, y_val, transforms_fn=val_transforms)
        train_cal_set  = ClassificationDataset(X_train_cal, y_train_cal, transforms_fn=val_transforms)

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading CIFAR10-Long-Tail data for pre-training complete")      
        

class Cifar100Data(Dataset):    
    def __init__(self, kwargs, experiment=None, name='cifar100', model_class=None):
        self.name = name
        self.model_class = model_class
        if experiment == 'pre-train':    
            kwargs.class_freqs = [1/kwargs.num_classes]*kwargs.num_classes
            self.generatePretrainingCifar100Data(
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition' or experiment == 'quantize'  or experiment == 'replicate':
            kwargs.dataset.class_freqs = [1/kwargs.dataset.num_classes]*kwargs.dataset.num_classes
            if kwargs.calibrator_version == 'v2':
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
            else:
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 
        print("Loading synthetic data for calibration complete")   

    def generatePretrainingCifar100Data(self, 
                                batch_size,
                                random_state):  
        data_dir =  f"./data/{self.name.upper()}"                     
        os.makedirs(data_dir, exist_ok=True)
        generator = torch.Generator().manual_seed(random_state)
        
        l_transform = transforms.Compose([
             transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(                        # use ImageNet mean/std for ViT
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        # l_transform = transforms.Compose([ 
        #         transforms.Resize((224, 224)),                          
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        #         ),
        # ])
        # train_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),               # match ViT input size
        #     transforms.RandomHorizontalFlip(p=0.5),      # common for CIFAR
        #     transforms.RandomCrop(224, padding=4),       # adds spatial jitter
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # brightness, contrast, saturation, hue
        #     transforms.RandomRotation(15),               # small rotations
        #     transforms.RandomGrayscale(p=0.1),           # sometimes grayscale
        #     transforms.ToTensor(), 
        #     transforms.Normalize(                        # use ImageNet mean/std for ViT
        #         mean=[0.485, 0.456, 0.406], 
        #         std=[0.229, 0.224, 0.225]
        #     ),                       
        # ])
        # val_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(), 
        #     transforms.Normalize(                        # use ImageNet mean/std for ViT
        #         mean=[0.485, 0.456, 0.406], 
        #         std=[0.229, 0.224, 0.225]
        #     ),            
        # ])
        
        full_train = datasets.CIFAR100(root=data_dir, train=True, 
                                        transform=l_transform, download=True)
        
        train_size = int(0.45 * len(full_train))
        val_size = int(0.1 * len(full_train))
        eval_cal_size = len(full_train) - train_size - val_size
        train_set, val_set, eval_cal_set = random_split(full_train, [train_size, val_size, eval_cal_size], generator=generator)

        train_cal_set = datasets.CIFAR100(root=data_dir, train=False, 
                                        transform=l_transform, download=True)
        
        print(f'Train shape: {len(train_set)}, Learn Calibration shape: {len(train_cal_set)}, Validation shape: {len(val_set)}, Eval Calibration shape: {len(eval_cal_set)}')

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading CIFAR100 data for pre-training complete") 

class Food101Datav2(Dataset):    
    def __init__(self, kwargs, experiment=None, name='food101'):       
        self.name = name   
        if experiment == 'pre-train':    
            kwargs.class_freqs = [1/kwargs.num_classes]*kwargs.num_classes
            self.generatePretrainingFood101Data(
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition' or experiment == 'quantize'  or experiment == 'replicate':
            kwargs.dataset.class_freqs = [1/kwargs.dataset.num_classes]*kwargs.dataset.num_classes
            if kwargs.calibrator_version == 'v2':
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
            else:
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 
        print("Loading synthetic data for calibration complete")   

    def generatePretrainingFood101Data(self, 
                                batch_size,
                                random_state):  
        data_dir =  f"./data/{self.name.upper()}"                     
        os.makedirs(data_dir, exist_ok=True)
        generator = torch.Generator().manual_seed(random_state)
        
        l_transform = transforms.Compose([
             transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(                        # use ImageNet mean/std for ViT
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])

        full_train = datasets.Food101(root=data_dir, split='train', 
                                        transform=l_transform, download=True)
        
        train_size = int(0.6 * len(full_train)) # 45.150
        val_size = int(0.1 * len(full_train)) # 4.515
        eval_cal_size = len(full_train) - train_size - val_size # 30.100
        train_set, val_set, eval_cal_set = random_split(full_train, [train_size, val_size, eval_cal_size], generator=generator)

        train_cal_set = datasets.Food101(root=data_dir, split='test', # 25.250
                                        transform=l_transform, download=True)
        
        print(f'Train shape: {len(train_set)}, Learn Calibration shape: {len(train_cal_set)}, Validation shape: {len(val_set)}, Eval Calibration shape: {len(eval_cal_set)}')

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading Food101 data for pre-training complete") 


def Cifar10OODData():
    pass

def Cifar100LongTailData():
    pass

def ImagenetData():
    pass

def ImagenetOODData():
    pass

def ImagenetLongTailData():
    pass


class CUbicCalibrationDataset(Dataset):
    def __init__(self, feats, logits, pca, y, p, num_classes, transforms_fn=None):
        self.num_classes = num_classes
        self.transforms_fn = transforms_fn
        self.feats = torch.tensor(feats, dtype=torch.float32)
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.pca = torch.tensor(pca, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.p = torch.tensor(p, dtype=torch.long)
        self.y_onehot = F.one_hot(self.y, num_classes=self.num_classes).float()
        self.p_onehot = F.one_hot(self.p, num_classes=self.num_classes).float()

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        feats = self.transforms_fn(self.feats[idx]) if self.transforms_fn is not None else self.feats[idx]
        logits = self.logits[idx]
        pca = self.pca[idx]
        y_onehot = self.y_onehot[idx]
        p_onehot = self.p_onehot[idx]
        p = self.p[idx]
        return feats, logits, pca, y_onehot, p, p_onehot


class CubicData:
    """
    Synthetic multiclass calibration dataset:
      p_model = softmax(z)
      q_true  = softmax(log p_model + gamma*(log p_model)^3)
      y ~ Cat(q_true)

    Returns loaders compatible with your existing CalibrationDatasetv2 pipeline.
    """
    def __init__(
        self, kwargs, num_workers=8, pin_memory=True):
        # cal_n=10_000,
        # test_n=25_000,
        # C=10,
        # gamma=0.15,
        # seed=42,
        # feature_dim=2048,
        # pca_dim=50,
        # batch_size=64,
            
        self.cal_n = kwargs.dataset.cal_n
        self.test_n = kwargs.dataset.test_n
        self.C = kwargs.dataset.num_classes
        self.gamma = kwargs.dataset.gamma
        self.scale = kwargs.dataset.scale
        self.seed = kwargs.seed
        self.warp_type = kwargs.dataset.warp_type

        self.feature_dim = kwargs.dataset.feature_dim
        self.pca_dim = kwargs.similarity_dim
        self.batch_size = kwargs.dataset.batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.make_loaders(val_fraction=0.1)
        
    @staticmethod
    def softmax_np(z, axis=1):
        z = z - z.max(axis=axis, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=axis, keepdims=True)

    def _make_split(self, n, seed):
        rng = np.random.default_rng(seed)
        
        # model logits -> model probs        
        z = (self.scale * rng.normal(size=(n, self.C))).astype(np.float32)
        p_model = self.softmax_np(z, axis=1).astype(np.float32)

        # cubic warp in log-prob space
        logp = np.log(np.clip(p_model, 1e-12, 1.0)).astype(np.float32)
        if self.warp_type == 'cubic':
            u = logp + self.gamma * (logp ** 3)     
        elif self.warp_type == 'quadratic':
            u = logp + self.gamma * logp * np.absolute(logp)
        elif self.warp_type == 'sinusoidal':   
            b = 3.0                 # frequency
            u = logp + self.gamma * np.sin(b * logp)
        else:
            raise ValueError(f"Unsupported warp type: {self.warp_type}. Choose 'cubic' or 'sinusoidal'.")
        q_true = self.softmax_np(u, axis=1).astype(np.float32)

        # labels from q_true
        y = np.array([rng.choice(self.C, p=q_true[i]) for i in range(n)], dtype=np.int64)

        # predicted class from model probs (like "preds" column)
        preds = np.argmax(p_model, axis=1).astype(np.int64)

        # --- Provide "feats" and "pca" to satisfy your dataset signature ---
        # If you want *only* logits/probs to matter, keep these as noise or zeros.
        feats = np.zeros((n, self.feature_dim), dtype=np.float32) #rng.normal(size=(n, self.feature_dim)).astype(np.float32)
        pca = np.zeros((n, self.pca_dim), dtype=np.float32) #rng.normal(size=(n, self.pca_dim)).astype(np.float32)

        # Return arrays: feats, logits, pca, y, preds
        # IMPORTANT: logits should be z (so downstream uses softmax(logits) = p_model)
        return feats, z, pca, y, preds

    def make_loaders(self, val_fraction=0.1):
        # "train calibration" split
        feats_train, logits_train, pca_train, y_train, p_train = self._make_split(self.cal_n, self.seed)

        # "eval calibration" split -> then split into test/val (90/10)
        feats_eval, logits_eval, pca_eval, y_eval, p_eval = self._make_split(self.test_n, self.seed + 10)

        (feats_test, feats_val,
         logits_test, logits_val,
         pca_test, pca_val,
         y_test, y_val,
         p_test, p_val) = train_test_split(
            feats_eval,
            logits_eval,
            pca_eval,
            y_eval,
            p_eval,
            test_size=val_fraction,
            random_state=self.seed,
            shuffle=True
        )

        # Build datasets
        train_set = CUbicCalibrationDataset(feats_train, logits_train, pca_train, y_train, p_train, num_classes=self.C)
        test_set  = CUbicCalibrationDataset(feats_test,  logits_test,  pca_test,  y_test,  p_test,  num_classes=self.C)
        val_set   = CUbicCalibrationDataset(feats_val,   logits_val,   pca_val,   y_val,   p_val,   num_classes=self.C)

        # DataLoaders
        self.data_train_cal_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        self.data_test_cal_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        self.data_val_cal_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

        return self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader

def find_categorical_columns(train_df, target_col='fact_cwsm_class', threshold_for_categorical=20):             

    # obvious types
    categorical_cols = train_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = train_df.select_dtypes(include=['number']).columns.tolist()

    # remove target
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    # heuristic: numeric columns with few unique values are likely categorical
    numeric_as_categorical = []
    for col in numerical_cols:
        n_unique = train_df[col].nunique(dropna=True)
        if n_unique <= threshold_for_categorical:
            numeric_as_categorical.append(col)

    # move them
    for col in numeric_as_categorical:
        numerical_cols.remove(col)
        categorical_cols.append(col)

    # optional: convert categorical columns to integer codes
    for col in categorical_cols:
        train_df[col] = train_df[col].astype('category')

    category_counts = [train_df[col].nunique() for col in categorical_cols]    

    # print("Categorical columns:", categorical_cols, '\n')
    # print("Numerical columns:", numerical_features)
    # print("Category counts:", category_counts)
    
    return categorical_cols, numerical_cols, category_counts 

def filter_unseen_rows(df, categorical_features, cat_levels):
    mask = pd.Series(True, index=df.index)

    for col in categorical_features:
        allowed = cat_levels[col]
        col_mask = df[col].isin(allowed)

        # if you want to keep NaNs too, use:
        # col_mask = df[col].isin(allowed) | df[col].isna()

        mask &= col_mask

    dropped = (~mask).sum()
    print(f"Dropping {dropped} rows with unseen categorical values out of {len(df)}")

    return df.loc[mask].copy()