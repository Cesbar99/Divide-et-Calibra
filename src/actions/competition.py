from competitors.competitors import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from algorithms.networks.networks import *
from algorithms.trainers.trainers import *
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict, OmegaConf
import time
from utils.utils import *
from data_sets.dataset import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from actions.test import test
from calibrator.cal_trainer import *
from calibrator.local_net import *
import optuna
from optuna.samplers import NSGAIISampler
from hp_opt.hp_opt import *
from pytorch_lightning.loggers import WandbLogger
import json
from tqdm import tqdm


def competition(kwargs, wandb_logger=None):
    seed = kwargs.seed    
    cuda_device = kwargs.cuda_device
    pl.seed_everything(seed, workers=True)  
    
    if kwargs.data == 'synthetic':
        dataset = SynthData(kwargs, experiment=kwargs.exp_name)  
    elif kwargs.data == 'covtype':
        dataset = CovTypeData(kwargs, experiment=kwargs.exp_name)  
    elif kwargs.data == 'otto':
        dataset = OttoData(kwargs, experiment=kwargs.exp_name)                    
    elif kwargs.data == 'mnist':
        if kwargs.dataset.variant:
            kwargs.data = kwargs.data + '_' + kwargs.dataset.variant                        
        dataset = MnistData(kwargs, experiment=kwargs.exp_name)
    elif kwargs.data == 'tissue':
        corrupt = kwargs.corruption_type
        kwargs.corruption_type = None
        dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)   
    elif kwargs.data == 'path':
        dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)       
    elif kwargs.data == 'cifar10':
        corrupt = kwargs.corruption_type
        kwargs.corruption_type = None
        dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)
    elif kwargs.data == 'cifar10_ood':
        dataset = Cifar10OODData(calibration=kwargs.calibration)
    elif kwargs.data == 'cifar10LT':
        dataset = Cifar10LongTailData(kwargs, experiment=kwargs.exp_name)
    elif kwargs.data == 'cifar100':
        corrupt = kwargs.corruption_type
        kwargs.corruption_type = None
        dataset = Cifar100Data(kwargs, experiment=kwargs.exp_name)  
    elif kwargs.data == 'weather':
        corrupt = False
        kwargs.corruption_type = None
        dataset = WeatherData(kwargs, experiment=kwargs.exp_name)        
    elif kwargs.data == 'food101':
        corrupt = kwargs.corruption_type
        kwargs.corruption_type = None
        dataset = Food101Datav2(kwargs, experiment=kwargs.exp_name)  
    elif kwargs.data == 'cifar100_longtail':
        dataset = Cifar100LongTailData(calibration=kwargs.calibration)
    elif kwargs.data == 'Imagenet':
        dataset = ImagenetData(calibration=kwargs.calibration)
    elif kwargs.data == 'imagenet_ood':
        dataset = ImagenetOODData(calibration=kwargs.calibration)
    elif kwargs.data == 'imagenet_longtail':
        dataset = ImagenetLongTailData(calibration=kwargs.calibration)    

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
    
    epochs = kwargs.checkpoint.epochs
    if epochs == 9:
        model_class = 'resnet'
    elif kwargs.checkpoint.epochs == 5:
        model_class = 'vit'
    elif kwargs.checkpoint.epochs == 20:
        model_class = 'convnext'
    else: # ftt uses 50 
        model_class = 'ftt'
        if not kwargs.data == 'weather':
            raise ValueError(
                f'Checkpoint not corresponding to a trained modl! {kwargs.checkpoint.epochs} was given but only 9 and 20 are supported')                    
            
    if 'SMS' in kwargs.methods:
        kwargs.method = 'SMS' # STRUCTURED MATRIX SCALING
        #num_classes}_classes_{kwargs.dataset.num_features}_features/"
        #os.makedirs(path, exist_ok=True) 
        os.makedirs(f"results/{kwargs.exp_name}_{kwargs.method}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
        raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class           
                )
        raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_{}.csv".format(
                kwargs.exp_name,
                kwargs.method,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                kwargs.models.max_iter,
                model_class                      
            )
        if kwargs.data == 'weather' and kwargs.dataset.shift: 
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_shift_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class           
                )
        # Assume you already trained `model`
        scaler = SMS()
        # Fit on validation set
        scaler.fit(dataset.data_train_cal_loader, device=cuda_device)
        
        raws = []
        # scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting SMS Calibration logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_test_cal, index=False)
        
        raws = []
        # scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting SMS Calibration logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_train_cal, index=False)
        
        if corrupt:            
            kwargs.corruption_type = corrupt
            if kwargs.data == 'tissue':
                dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)                
            elif kwargs.data == 'cifar10':                                
                dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)       
            elif kwargs.data == 'cifar100':
                dataset = Cifar100Data(kwargs, experiment=kwargs.exp_name)                 
                
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.corruption_type,
                        seed,
                        kwargs.models.max_iter,
                        model_class           
                    )
            raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.corruption_type,
                    seed,
                    kwargs.models.max_iter,
                    model_class                      
                )
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_test_cal, index=False)
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_train_cal, index=False)
        
        print(f"\nSTART TESTING {kwargs.method}!")        
        test(kwargs)
    
    if 'DC' in kwargs.methods:
        kwargs.method = 'DC' #DIRICHLET CALIBRATION
        #num_classes}_classes_{kwargs.dataset.num_features}_features/"
        #os.makedirs(path, exist_ok=True) 
        os.makedirs(f"results/{kwargs.exp_name}_{kwargs.method}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
        raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class           
                )
        raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_{}.csv".format(
                kwargs.exp_name,
                kwargs.method,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                kwargs.models.max_iter,
                model_class                      
            )
        if kwargs.data == 'weather' and kwargs.dataset.shift: 
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_shift_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class           
                )
            
        # Assume you already trained `model`
        scaler = DirichletCalibrator(n_classes=kwargs.dataset.num_classes, max_iter=kwargs.models.max_iter, lr=kwargs.models.temp_lr)
        # Fit on validation set
        scaler.fit(dataset.data_train_cal_loader, device=cuda_device)
        
        raws = []
        scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting Dirichlet Calibration logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_test_cal, index=False)
        
        raws = []
        scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting Dirichlet Calibration logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_train_cal, index=False)
        
        if corrupt:            
            kwargs.corruption_type = corrupt
            if kwargs.data == 'tissue':
                dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)                
            elif kwargs.data == 'cifar10':                                
                dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)       
            elif kwargs.data == 'cifar100':
                dataset = Cifar100Data(kwargs, experiment=kwargs.exp_name)                 
                
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.corruption_type,
                        seed,
                        kwargs.models.max_iter,
                        model_class           
                    )
            raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.corruption_type,
                    seed,
                    kwargs.models.max_iter,
                    model_class                      
                )
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_test_cal, index=False)
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_train_cal, index=False)                
        
        print(f"\nSTART TESTING {kwargs.method}!")        
        test(kwargs)
    
    if 'TS' in kwargs.methods:
        kwargs.method = 'TS' #TEMPERATURE SCALING
        #num_classes}_classes_{kwargs.dataset.num_features}_features/"
        #os.makedirs(path, exist_ok=True) 
        os.makedirs(f"results/{kwargs.exp_name}_{kwargs.method}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
        raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class           
                )
        raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_{}.csv".format(
                kwargs.exp_name,
                kwargs.method,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                kwargs.models.max_iter,
                model_class                      
            )
        if kwargs.data == 'weather' and kwargs.dataset.shift: 
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_shift_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class           
                )
            
        # Assume you already trained `model`
        scaler = TemperatureScaler(kwargs.models.max_iter, kwargs.models.temp_lr)
        # Fit on validation set
        scaler.fit(dataset.data_train_cal_loader, device=cuda_device)
        
        raws = []
        scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting Temeperature scaling logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_test_cal, index=False)
        
        raws = []
        scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting Temeperature Scaling logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_train_cal, index=False)
        
        if corrupt:            
            kwargs.corruption_type = corrupt
            
            if kwargs.data == 'tissue':
                dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)                
            elif kwargs.data == 'cifar10':                                
                dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)       
            elif kwargs.data == 'cifar100':
                dataset = Cifar100Data(kwargs, experiment=kwargs.exp_name)                 
                
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.corruption_type,
                        seed,
                        kwargs.models.max_iter,
                        model_class           
                    )
            raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.corruption_type,
                    seed,
                    kwargs.models.max_iter,
                    model_class                      
                )
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_test_cal, index=False)
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_train_cal, index=False)       
            
        if corrupt:            
            kwargs.corruption_type = corrupt
            
            if kwargs.data == 'tissue':
                dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)                
            elif kwargs.data == 'cifar10':                                
                dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)       
            elif kwargs.data == 'cifar100':
                dataset = Cifar100Data(kwargs, experiment=kwargs.exp_name)                 
                
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.corruption_type,
                        seed,
                        kwargs.models.max_iter,
                        model_class           
                    )
            raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.corruption_type,
                    seed,
                    kwargs.models.max_iter,
                    model_class                      
                )            
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_test_cal, index=False)
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_train_cal, index=False)                                 
        
        print(f"\nSTART TESTING {kwargs.method}!")        
        test(kwargs)

    if 'IR' in kwargs.methods:    
        kwargs.method = 'IR' #ISOTONIC REGRESSION
        #num_classes}_classes_{kwargs.dataset.num_features}_features/"
        #os.makedirs(path, exist_ok=True) 
        os.makedirs(f"results/{kwargs.exp_name}_{kwargs.method}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
        raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class           
                )
        raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_{}.csv".format(
                kwargs.exp_name,
                kwargs.method,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                kwargs.models.max_iter,
                model_class                      
            )
        if kwargs.data == 'weather' and kwargs.dataset.shift: 
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_shift_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class           
                )
            
        # Assume you already trained `model`
        scaler = IsotonicCalibrator(out_dim=kwargs.dataset.num_classes)
        # Fit on validation set
        scaler.fit(dataset.data_train_cal_loader, device=cuda_device)
        
        raws = []        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        with torch.no_grad():
            for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting Isotonic Regression logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_test_cal, index=False)
        
        raws = []        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting Isotonic Regression logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_train_cal, index=False)
        
        if corrupt:            
            kwargs.corruption_type = corrupt
            
            if kwargs.data == 'tissue':
                dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)                
            elif kwargs.data == 'cifar10':                                
                dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)       
            elif kwargs.data == 'cifar100':
                dataset = Cifar100Data(kwargs, experiment=kwargs.exp_name)                 
                
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.corruption_type,
                        seed,
                        kwargs.models.max_iter,
                        model_class           
                    )
            raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.corruption_type,
                    seed,
                    kwargs.models.max_iter,
                    model_class                      
                )
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_test_cal, index=False)
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_train_cal, index=False)                
        
        
        print(f"\nSTART TESTING {kwargs.method}!")        
        test(kwargs)

    if 'PS' in kwargs.methods:    
        kwargs.method = 'PS' #PLATT SCALING
        #num_classes}_classes_{kwargs.dataset.num_features}_features/"
        #os.makedirs(path, exist_ok=True) 
        os.makedirs(f"results/{kwargs.exp_name}_{kwargs.method}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
        raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class           
                )
        raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_{}.csv".format(
                kwargs.exp_name,
                kwargs.method,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                kwargs.models.max_iter,
                model_class                      
            )
        if kwargs.data == 'weather' and kwargs.dataset.shift: 
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_shift_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class           
                )
            
        # Assume you already trained `model`
        scaler = PlattScaler(out_dim=kwargs.dataset.num_classes)
        # Fit on validation set
        scaler.fit(dataset.data_train_cal_loader, device=cuda_device)
        
        raws = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        with torch.no_grad():
            for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting Platt scaling logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_test_cal, index=False)
        
        raws = []        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting Platt Scaling logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_train_cal, index=False)
        
        if corrupt:            
            kwargs.corruption_type = corrupt
            
            if kwargs.data == 'tissue':
                dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)                
            elif kwargs.data == 'cifar10':                                
                dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)       
            elif kwargs.data == 'cifar100':
                dataset = Cifar100Data(kwargs, experiment=kwargs.exp_name)                 
                
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.corruption_type,
                        seed,
                        kwargs.models.max_iter,
                        model_class           
                    )
            raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.corruption_type,
                    seed,
                    kwargs.models.max_iter,
                    model_class                      
                )
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_test_cal, index=False)
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_train_cal, index=False)                
                    
        
        print(f"\nSTART TESTING {kwargs.method}!")        
        test(kwargs)
    

    if 'PC' in kwargs.methods: # ProCal        
        kwargs.method = 'PC' # STRUCTURED MATRIX SCALING
        #num_classes}_classes_{kwargs.dataset.num_features}_features/"
        #os.makedirs(path, exist_ok=True) 
        os.makedirs(f"results/{kwargs.exp_name}_{kwargs.method}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
        raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class            
                )
        raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_{}.csv".format(
                kwargs.exp_name,
                kwargs.method,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                kwargs.models.max_iter,
                model_class                      
            )
        if kwargs.data == 'weather' and kwargs.dataset.shift: 
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_shift_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter,
                    model_class           
                )
            
        # Assume you already trained `model`
        scaler = DensityRatioCalibration(num_neighbors=kwargs.models.num_neighbors)
        # Fit on validation set
        scaler.fit(dataset.data_train_cal_loader, device=cuda_device)
        
        raws = []
        # scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting PC Calibration logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_test_cal, index=False)
        
        raws = []
        # scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting PC Calibration logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res, pca = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_train_cal, index=False)
        
        if corrupt:            
            kwargs.corruption_type = corrupt
            if kwargs.data == 'tissue':
                dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)                
            elif kwargs.data == 'cifar10':                                
                dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)       
            elif kwargs.data == 'cifar100':
                dataset = Cifar100Data(kwargs, experiment=kwargs.exp_name)                 
                
            raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.corruption_type,
                        seed,
                        kwargs.models.max_iter,
                model_class
                    )
            raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.corruption_type,
                    seed,
                    kwargs.models.max_iter,
                    model_class                      
                )
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_test_cal, index=False)
            
            raws = []
            # scaler.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # scaler.to(device)

            with torch.no_grad():
                for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting SMS Calibration logits"):
                    batch = [b.to(device) for b in batch]                
                    raw = scaler.calibrated_predictions(batch)
                    raws.append(raw)
                    
            res, pca = get_raw_res(raws, features=True, reduced_dim=None)
            res.to_csv(raw_results_path_train_cal, index=False)
        
        print(f"\nSTART TESTING {kwargs.method}!")        
        test(kwargs)
    
    
    
    