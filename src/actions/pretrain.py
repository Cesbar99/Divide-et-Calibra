import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import random as pyrandom
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
from tqdm import tqdm
import optuna
from optuna.samplers import NSGAIISampler
from hp_opt.hp_opt import *
from imagecorruptions import corrupt #corrupt_batch

def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    elif isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    else:
        return x

def corrupt_batch(batch, corruption_name, severity):
    return np.stack([
        corrupt(img, corruption_name=corruption_name, severity=severity)
        for img in batch
    ])

def pretrain(kwargs, wandb_logger):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = kwargs.seed
    #pl.seed_everything(seed, workers=True)  
    # if kwargs.data != 'food101':
    #     total_epochs = kwargs.models.epochs
    #     temperature = kwargs.models.temperature
    # else:
    #     total_epochs = 'None'
    #     temperature = 1.0
    total_epochs = kwargs.models.epochs
    if kwargs.corruption_type:
        total_epochs = kwargs.checkpoint.epochs
    temperature = kwargs.models.temperature
    cuda_device = kwargs.cuda_device
    epochs = kwargs.checkpoint.epochs
    print(epochs)
    if epochs == 9:
        model_class = 'resnet'
    elif kwargs.checkpoint.epochs == 5:
        model_class = 'vit'
    elif kwargs.checkpoint.epochs == 20:
        model_class = 'convnext'
    else:
        model_class = 'ftt'
        if not kwargs.data == 'weather':
            raise ValueError(
                f'Checkpoint not corresponding to a trained modl! {kwargs.checkpoint.epochs} was given but only 9 and 20 are supported')
    
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
        "frost", # good severity: 1     
        "brightness", # good severity: 3 for cifar10, 1 for cifar100
        "contrast",
        "pixelate",  # good severity: 1       
    ]

    sev = kwargs.severity
    if kwargs.corruption_type:
        corruption_text = kwargs.corruption_type
        corruption_name = kwargs.corruption_type
        if sev > 0:
            severity = kwargs.severity
            corruption_text += f'_severity_{severity}'
        print("CORRUPTION TEXT: ", corruption_text)
    else:
        corruption_text = "None"
    if kwargs.data == 'synthetic':
        dataset = SynthData(kwargs.dataset, experiment=kwargs.exp_name)
        pl_model = SynthTab(input_dim=kwargs.dataset.num_features,            
                            output_dim=kwargs.dataset.num_classes,
                            temperature=kwargs.models.temperature,
                            optimizer_cfg=kwargs.models.optimizer,
                            use_acc=kwargs.models.use_acc
                        )
        
    elif kwargs.data == 'covtype':
        dataset = CovTypeData(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data)
        pl_model = CovTypeModel(kwargs.models, dataset.numerical_features, dataset.category_counts)
        
    elif kwargs.data == 'weather':
        dataset = WeatherData(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data, seed=seed)
        # splits = WeatherData(kwargs.dataset, experiment='xg_debug', name='weather', seed=42)

        # # number of classes
        # num_classes = len(set(splits.y_train))

        # model = XGBClassifier(
        #     objective='multi:softprob',
        #     num_class=num_classes,
        #     eval_metric='mlogloss',

        #     # good starting defaults
        #     n_estimators=500,
        #     learning_rate=0.05,
        #     max_depth=6,
        #     subsample=0.8,
        #     colsample_bytree=0.8,

        #     tree_method='hist',  # fast
        #     random_state=42,
        # )

        # # train with validation monitoring
        # model.fit(
        #     splits.X_train, splits.y_train,
        #     eval_set=[(splits.X_val, splits.y_val)],
        #     verbose=True
        # )

        # # predictions
        # y_pred = model.predict(splits.X_eval_cal)

        # # evaluation
        # acc = accuracy_score(splits.y_eval_cal, y_pred)
        # print(f"\nEval accuracy: {acc:.4f}")

        # print("\nClassification report:")
        # print(classification_report(splits.y_eval_cal, y_pred))
        
        pl_model = WeatherModel(kwargs.models, dataset.numerical_features, dataset.category_counts, dataset.class_counts)
    
    elif kwargs.data == 'otto':
        dataset = OttoData(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data)
        pl_model = OttoModel(kwargs.models, dataset.numerical_features)        
        
    elif kwargs.data == 'mnist':
        if kwargs.dataset.variant:
            kwargs.data = kwargs.data + '_' + kwargs.dataset.variant            
        dataset = MnistData(kwargs.dataset, experiment=kwargs.exp_name)
        pl_model = MnistModel(kwargs.models)
    
    elif kwargs.data == 'tissue':
        dataset = MedMnistData(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data, model_class=model_class)
        pl_model = MedMnistModel(kwargs.models)    
        
    elif kwargs.data == 'path':
        dataset = MedMnistData(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data)
        pl_model = MedMnistModel(kwargs.models)
        
    elif kwargs.data == 'cifar10':
        dataset = Cifar10Data(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data, model_class=model_class)
        # dataset2 = Cifar10Data(kwargs.dataset, experiment='calibrate', name=kwargs.data)
        pl_model = Cifar10Model(kwargs.models)
        
    elif kwargs.data == 'cifar10LT':
        dataset = Cifar10LongTailData(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data)
        pl_model = Cifar10LongTailModel(kwargs.models)
    
    elif kwargs.data == 'cifar10_ood':
        dataset = Cifar10OODData()
        pl_model = Cifar10OODModel()
        
    elif kwargs.data == 'cifar100':
        dataset = Cifar100Data(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data, model_class=model_class)
        # dataset2 = Cifar100Data(kwargs.dataset, experiment='calibrate', name=kwargs.data)
        pl_model = Cifar100Model(kwargs.models)   
        
    elif kwargs.data == 'food101':
        # dataset = Food101Data(kwargs, experiment=kwargs.exp_name, name=kwargs.data) # generatefoodDataforPca(kwargs)        
        dataset = Food101Datav2(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data) # generatefoodDataforPca(kwargs)        
        pl_model = Food101Model(kwargs.models)   
        
    elif kwargs.data == 'cifar100_longtail':
        dataset = Cifar100LongTailData()
        pl_model = Cifar100LongTailModel()
        
    elif kwargs.data == 'Imagenet':
        dataset = ImagenetData()
        pl_model = ImagenetModel()
        
    elif kwargs.data == 'imagenet_ood':
        dataset = ImagenetOODData()
        pl_model = ImagenetOODModel()
        
    elif kwargs.data == 'imagenet_longtail':
        dataset = ImagenetLongTailData()  
        pl_model = ImagenetLongTailModel()    
        
    path = f"checkpoints/{kwargs.exp_name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features/"    
    os.makedirs(path, exist_ok=True) 
    os.makedirs(f"results/{kwargs.exp_name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
    path_model = "checkpoints/{}/{}_{}_classes_{}_features/classifier_seed-{}_ep-{}_tmp_{}.pt".format(
            kwargs.exp_name,
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            seed,
            total_epochs,
            temperature
        )
    if kwargs.corruption_type:
        raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                corruption_text,
                seed,
                total_epochs,
                temperature            
            )
        raw_results_path_eval_cal = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                corruption_text,
                seed,
                total_epochs,
                temperature            
            )
        raw_results_path_clean_eval_cal = "results/{}/{}_{}_classes_{}_features/raw_results_clean_eval_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                corruption_text,
                seed,
                total_epochs,
                temperature            
            )
    else:
        raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                total_epochs,
                temperature            
            )
        raw_results_path_eval_cal = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                total_epochs,
                temperature            
            )
    # if kwargs.data == 'food101':
    #     raw_results_path_val_cal = "results/{}/{}_{}_classes_{}_features/raw_results_val_cal_seed-{}_ep-{}_tmp_{}.csv".format(
    #         kwargs.exp_name,
    #         kwargs.data,
    #         kwargs.dataset.num_classes,
    #         kwargs.dataset.num_features,
    #         seed,
    #         total_epochs,
    #         temperature            
    #     )
    
    # if kwargs.data != 'food101':
    if (kwargs.corruption_type) or (kwargs.extract_embeddings):
        if kwargs.data == 'weather':
            best_model_path = path + f"classifier_seed-{seed}_ep-{kwargs.checkpoint.epochs}_tmp_{kwargs.models.temperature}.pt.ckpt"                        # Static filename (no epoch suffix)                
            checkpoint = torch.load(best_model_path, map_location=device)     
            state_dict = checkpoint["state_dict"]            
            pl_model.load_state_dict(state_dict)
        else:
            if kwargs.checkpoint.epochs in [5, 20, 50]:
                best_model_path = path + f"classifier_seed-{seed}_ep-{kwargs.checkpoint.epochs}_tmp_{kwargs.models.temperature}.pt.ckpt"                        # Static filename (no epoch suffix)                
                checkpoint = torch.load(best_model_path, map_location=device)
                #pl_model.model.load_state_dict(checkpoint)
                state_dict = checkpoint["state_dict"]     
                new_state_dict = {}
                # for k, v in state_dict.items():
                #     if k.startswith("model."):
                #         new_state_dict[k[len("model."):]] = v
                #     else:
                #         new_state_dict[k] = v
                # pl_model.load_state_dict(new_state_dict)       
                pl_model.load_state_dict(state_dict)            
            elif kwargs.checkpoint.epochs == 9:            
                best_model_path = path + f"classifier_seed-{seed}_ep-{kwargs.checkpoint.epochs}_tmp_{kwargs.models.temperature}.pt"                        # Static filename (no epoch suffix)                
                checkpoint = torch.load(best_model_path, map_location=device)            
                if kwargs.data == 'tissue':
                    fixed_checkpoint = {}
                    for k, v in checkpoint.items():
                        if k.startswith("resnet50.fc.1."):
                            fixed_checkpoint[k.replace("resnet50.fc.1.", "resnet50.fc.0.")] = v
                        else:
                            fixed_checkpoint[k] = v
                else:
                    fixed_checkpoint = checkpoint
                pl_model.model.load_state_dict(fixed_checkpoint)
                # pl_model.model.load_state_dict(checkpoint)
    else:      
        if kwargs.use_optuna:     
            csv_path = f"optuna_logs/optuna_best_configs_pretrain_vit_{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features.csv"
            
            if seed == 42: # only run optuna for one seed to save time
                print(f'STARTING OPTUNA HYPERPARAMETER SEARCH FOR {kwargs.n_trials} TRIALS OF {kwargs.optuna_epochs} EPOCHS PRE-TRAINING...\n')       
                
                # search_space = {
                # "lin_comb": [0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99],
                # "ceiling": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
                # "feature_dim": [8, 10, 16, 32, 64, 128]
                # }   
                # sampler = optuna.samplers.GridSampler(search_space)
                study = optuna.create_study(direction="maximize",study_name="hyper_params_pretrain", # "minimize"
                                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
                calls = [print_callback]
                study.optimize(
                    lambda trial: pretrain_objective(trial, kwargs, dataset.data_train_loader, 
                                                    dataset.data_val_loader, wandb_logger),
                    n_trials=kwargs.n_trials,
                    show_progress_bar=True,
                    callbacks=calls
                )
                
                # Print best result
                print("Best trial:")
                print(f"  Value: {study.best_trial.value}")
                for key, value in study.best_trial.params.items():
                    print(f"    {key}: {value}")
                    kwargs.models.optimizer[key] = value   
                    
                # ---- SAVE BEST CONFIG TO CSV ----            

                file_exists = os.path.isfile(csv_path)

                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "study_name",                        
                            "value",                        
                            "optuna_epochs",
                            "train_epochs",
                            "name",
                            "lr",
                            "weight_decay"                      
                        ]
                    )

                    # write header only once
                    if not file_exists:
                        writer.writeheader()

                    writer.writerow({
                        "study_name": study.study_name,                                                            
                        "value": study.best_trial.value,                                                                                
                        "optuna_epochs": kwargs.optuna_epochs,
                        "train_epochs": kwargs.models.epochs,
                        "name": study.best_trial.params["name"], # kwargs.models.optimizer,
                        "lr": study.best_trial.params["lr"],
                        "weight_decay": study.best_trial.params["weight_decay"]                
                    })

                print(f"Saved best hyperparameters to {csv_path}")  
                        
            else:
                print("Loading hyperparameters from CSV...")
                load_optuna_config(csv_path, kwargs.models.optimizer, pretrain=True)            
                    
                    
        ############## FITTING TIME ##############
        if kwargs.data == 'cifar10':
            pl_model = Cifar10Model(kwargs.models)   
        elif kwargs.data == 'cifar100':
            pl_model = Cifar100Model(kwargs.models)   
        elif kwargs.data == 'tissue':
            pl_model = MedMnistModel(kwargs.models)           
        print(F'BEGIN PRE-TRAINING FOR {total_epochs} EPOCHS WITH SEED {seed} AND {kwargs.models.temperature} TEMPERATURE!')        
        trainer = pl.Trainer(
                max_epochs=total_epochs,
                accelerator="cuda",
                devices=[cuda_device],
                logger=wandb_logger,
                check_val_every_n_epoch=1,            
                deterministic=True,
                callbacks=[ ClearCacheCallback(), 
                    EarlyStopping(
                         monitor="val_loss", #val_loss
                         patience=10,
                         mode="min", #"max"
                         verbose=True,
                         min_delta=0.0,
                    ),
                    ModelCheckpoint(
                        monitor="val_loss", # "val_loss",                                                                                             # Metric to track
                        mode="min", # "min"                                                                                                    # Lower is better
                        save_top_k=1,                                                                                                   # Only keep the best model
                        filename=f"classifier_seed-{seed}_ep-{total_epochs}_tmp_{kwargs.models.temperature}.pt",                        # Static filename (no epoch suffix)
                        dirpath=path,                                                                                                   # Save in your existing checkpoint folder
                        save_weights_only=True,                                                                                         # Save only weights (not full LightningModule)
                        auto_insert_metric_name=False,                                                                                  # Prevent metric name in filename
                        every_n_epochs=1,                                                                                               # Run every epoch                    
                        enable_version_counter=False,
                        verbose=True
                    ) ,                 
                ]
            )
        start = time.time()
        trainer.fit(pl_model, dataset.data_train_loader,
                        dataset.data_val_loader)
        train_time = time.time() - start
        # print(train_time)
        # torch.save(pl_model.model.state_dict(), path_model)
        best_model_path = trainer.checkpoint_callback.best_model_path
        #print(F'LOADING CHECKPOINT FILE {best_model_path}')
        #best_model_path = 'REPLACE'            
        
        checkpoint = torch.load(best_model_path)
        pl_model.load_state_dict(checkpoint['state_dict'])
    
    if kwargs.return_features:
        raws = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if kwargs.data != 'food101':
        pl_model.eval()        
        pl_model.to(device)
        shown = False

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting calibration features"):
                #batch = [b.to(device) for b in batch] 
                batch = move_to_device(batch, device)
                # if kwargs.data != 'food101':            
                raw = pl_model.extract_features(batch)
                raw_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in raw.items()}
                raws.append(raw_cpu)


        #all_features = torch.cat(all_features)
        print('features shape: ', raws[1]['features'].shape, raws[1]['preds'].shape, raws[1]['true'].shape)
        res, pca = get_raw_res(raws, features=True, reduced_dim=kwargs.similarity_dim) # fit_pca=pca
    else:
        raws = trainer.predict(pl_model, dataset.data_train_cal_loader) #dataset.data_train_cal_loader
        res, pca = get_raw_res(raws)
    res.to_csv(raw_results_path_train_cal, index=False)

    if kwargs.return_features:
        raws = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if kwargs.data != 'food101':
        pl_model.eval()
        pl_model.to(device)

        with torch.no_grad():
            # if kwargs.data != 'food101':  
            for batch in tqdm(dataset.data_eval_cal_loader, desc="Extracting test features"):
                #batch = [b.to(device) for b in batch]               
                batch = move_to_device(batch, device)
                                    
                raw = pl_model.extract_features(batch)
                raw_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in raw.items()}
                raws.append(raw_cpu)
                # raws.append(raw)

        print('features shape: ', raws[1]['features'].shape)
        res, pca = get_raw_res(raws, features=True, reduced_dim=kwargs.similarity_dim, fit_pca=pca) #fit_pca=pca
    else:
        raws = trainer.predict(pl_model, dataset.data_eval_cal_loader) #dataset.data_eval_cal_loader
        res, pca = get_raw_res(raws)
    res.to_csv(raw_results_path_clean_eval_cal, index=False)

    
    if kwargs.return_features and kwargs.corruption_type:
        raws = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if kwargs.data != 'food101':
        pl_model.eval()
        pl_model.to(device)

        with torch.no_grad():
            # if kwargs.data != 'food101':  
            for batch in tqdm(dataset.data_eval_cal_loader, desc="Extracting corrupted test features"):
                #batch = [b.to(device) for b in batch]               
                batch = move_to_device(batch, device)
                                    
                images = batch[0]
                # ---- APPLY CORRUPTION ----
                images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')                                        

                images_np = corrupt_batch(
                    images_np,
                    corruption_name=corruption_name, #"gaussian_noise",  # change as needed
                    severity=sev, #pyrandom.randint(1, 5)
                )

                images = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
                images = images.to(device)
                
                batch[0] = images
                                 
                raw = pl_model.extract_features(batch)
                raw_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in raw.items()}
                raws.append(raw_cpu)
                # raws.append(raw)
        print('features shape: ', raws[1]['features'].shape)
        res, pca = get_raw_res(raws, features=True, reduced_dim=kwargs.similarity_dim, fit_pca=pca) #fit_pca=pca

    res.to_csv(raw_results_path_eval_cal, index=False)
    
    if kwargs.data == 'weather':
        raw_results_path_eval_cal_shift = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_shift_seed-{}_ep-{}_tmp_{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                total_epochs,
                temperature            
            )
        if kwargs.return_features:
            raws = []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # if kwargs.data != 'food101':
            pl_model.eval()
            pl_model.to(device)

            with torch.no_grad():
                # if kwargs.data != 'food101':  
                for batch in tqdm(dataset.data_eval_cal_shift_loader, desc="Extracting features"):
                    #batch = [b.to(device) for b in batch]               
                    batch = move_to_device(batch, device)
                        
                    if kwargs.corruption_type:
                        images = batch[0]
                        # ---- APPLY CORRUPTION ----
                        images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')                                        

                        images_np = corrupt_batch(
                            images_np,
                            corruption_name=corruption_name, #"gaussian_noise",  # change as needed
                            severity=sev, #pyrandom.randint(1, 5)
                        )

                        images = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
                        images = images.to(device)
                        
                        batch[0] = images
                                    
                    raw = pl_model.extract_features(batch)
                    raw_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in raw.items()}
                    raws.append(raw_cpu)
                    # raws.append(raw)                
                        
            #all_raws = torch.cat(all_raws)
            print('features shape: ', raws[1]['features'].shape)
            res, pca = get_raw_res(raws, features=True, reduced_dim=kwargs.similarity_dim, fit_pca=pca) #fit_pca=pca
        else:
            raws = trainer.predict(pl_model, dataset.data_eval_cal_shift_loader) #dataset.data_eval_cal_loader
            res, pca = get_raw_res(raws)
        res.to_csv(raw_results_path_eval_cal_shift, index=False)


    print("PRE-TRAINING OVER!")
    print("START TESTING!")
    # kwargs.corruption_type = corruption_name
    test(kwargs)
    # kwargs.corruption_type = corruption_name
    
    
    
    
    
