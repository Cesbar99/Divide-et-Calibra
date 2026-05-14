import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from utils.utils import *
import matplotlib.ticker as mticker

def test(kwargs):
    class_freqs = kwargs.dataset.class_freqs
    #scaler = StandardScaler()    
    if kwargs.data == 'mnist' and kwargs.dataset.variant:
            kwargs.data = kwargs.data + '_' + kwargs.dataset.variant
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
    # if (kwargs.corruption_type) and (kwargs.corruption_type not in corruptions):
    #     raise ValueError(f'Unknown corruption type! {kwargs.corruption_type} was given.')
    sev = kwargs.severity
    if kwargs.corruption_type:
        corruption_text = kwargs.corruption_type
        if sev > 0:
            severity = kwargs.severity
            corruption_text += f'_severity_{severity}'
        print("CORRUPTION TEXT: ", corruption_text)
    else:
        corruption_text = "None"
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
            
    if kwargs.exp_name == 'pre-train':   
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            to_add = kwargs.data + '_' + 'shift' 
        else:
            to_add = kwargs.data
        # if kwargs.data != 'food101':        
        temperature = kwargs.checkpoint.temperature
        # else:
        #     epochs = 'None'
        #     temperature = 1.0    
                   
        gamma = kwargs.gamma            
        n_bins = kwargs.n_bins_calibration_metrics  
        n_bins_esse = kwargs.n_bins_esse
        appendix =  kwargs.exp_name + '_' + to_add + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
        test_file_name = 'multicalss_calibration_train_cal'+'.png'                
        cal_file_name = 'multicalss_calibration_eval_cal'+'.png'        
        save_path = join(kwargs.save_path_calibration_plots, appendix)
        os.makedirs(save_path, exist_ok=True)   
        if kwargs.corruption_type: 
            cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    corruption_text,
                    kwargs.seed,
                    epochs,
                    temperature            
                )
            
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    corruption_text,
                    kwargs.seed,
                    epochs,
                    temperature            
                )
        else:
            cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed,
                    epochs,
                    temperature            
                )
            if kwargs.data == 'weather' and kwargs.dataset.shift:
                test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_shift_seed-{}_ep-{}_tmp_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed,
                    epochs,
                    temperature            
                )
            else:
                test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed,
                    epochs,
                    temperature            
                )
        
        # Load your data
        df_cal = pd.read_csv(cal_results)
        df_test = pd.read_csv(test_results)
        
        # Extract features and labels
        cols = df_cal.columns
        # Single pass grouping
        features_cols = [c for c in cols if c.startswith("features")]
        logits_cols   = [c for c in cols if c.startswith("logits")]
        pca_cols      = [c for c in cols if c.startswith("pca")]
        # Extract values
        feats_train_cal  = df_cal[features_cols].values
        logits_train_cal = df_cal[logits_cols].values
        pca_train_cal    = df_cal[pca_cols].values

        y_train_cal = df_cal["true"].values
        p_train_cal = df_cal["preds"].values

        cols = df_test.columns
        # Single pass grouping
        features_cols = [c for c in cols if c.startswith("features")]
        logits_cols   = [c for c in cols if c.startswith("logits")]
        pca_cols      = [c for c in cols if c.startswith("pca")]
        # Extract values
        feats_eval_cal  = df_test[features_cols].values
        logits_eval_cal = df_test[logits_cols].values
        pca_eval_cal    = df_test[pca_cols].values
        
        y_eval_cal = df_test["true"].values
        p_eval_cal = df_test["preds"].values

        # if kwargs.data != 'food101':
        # Split into 90% test and 10% val
        if kwargs.corruption_type:
            feats_test, logits_test, pca_test, y_test, p_test = feats_eval_cal, logits_eval_cal, pca_eval_cal, y_eval_cal, p_eval_cal
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
        # Compute accuracy
        #accuracy_test = (df_test['preds'] == df_test['true']).mean()
        accuracy_test = (y_test == p_test).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')
        accs = {'acc': [accuracy_test]}
        df_accs = pd.DataFrame(accs)
        #accuracy_cal = (df_cal['preds'] == df_cal['true']).mean()
        accuracy_cal = (y_train_cal == p_train_cal).mean()
        print(f'Cal accuracy: {accuracy_cal:.2%}')   
        
        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"accs_eval_cal_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv")
        df_accs.to_csv(output_file, index=False) 
        
        if not kwargs.only_test: # COMPUTE METRIC IN ADDITION TO ACCURACY
        
            # Extract logits and true labels                    
            labels_test = y_test #df_test['true']
            #pca_test = df_test.filter(regex=r'^features')
            
            logits_cal = logits_train_cal #df_cal.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
            pca_cal = pca_train_cal #df_cal.filter(regex=r'^features')
            labels_cal = y_train_cal #df_cal['true']
            
            #logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
            logits_test_ = torch.tensor(logits_test, dtype=torch.float32)
            #pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
            pca_test_ = torch.tensor(pca_test, dtype=torch.float32)            
            #y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)
            y_true_test_ = torch.tensor(labels_test, dtype=torch.long)        

            #logits_cal_ = torch.tensor(logits_cal.values, dtype=torch.float32)
            logits_cal_ = torch.tensor(logits_cal, dtype=torch.float32)
            #pca_cal_ = torch.tensor(pca_cal.values, dtype=torch.float32)
            pca_cal_ = torch.tensor(pca_cal, dtype=torch.float32)
            #y_true_cal_ = torch.tensor(labels_cal.values, dtype=torch.long)
            y_true_cal_ = torch.tensor(labels_cal, dtype=torch.long)

            # Convert logits to probabilities
            probs_test = F.softmax(logits_test_, dim=1)
            probs_cal = F.softmax(logits_cal_, dim=1)

            # Compute calibration metrics
            ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
            results = {
                "ECCE": [ecce_test],       
                "ECE": [ece_test],
                "MCE": [mce_test],
                "Brier": [brier_test],
                "NLL": [nll_test],
                "LCE": [lce_test],
                "MLCE": [mlce_test]
            }

            # Convert to DataFrame
            df = pd.DataFrame(results)

            # Specify your directory and filename
            output_dir = join(kwargs.save_path_calibration_metrics, appendix)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"metric_eval_cal_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv")

            # Save to CSV
            df.to_csv(output_file, index=False)  
            
            # ---- Save aggregated ESS profile ----
            ess_results = {
                "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                "avg_ess": ess_profile["avg_ess_per_bin"],
                "count": ess_profile["count_per_bin"]
            }

            df_ess = pd.DataFrame(ess_results)

            ess_output_file = os.path.join(
                output_dir,
                f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
            )

            df_ess.to_csv(ess_output_file, index=False)
            print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")                
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name)
            #multiclass_calibration_plot(y_true_cal_, probs_cal, n_bins=n_bins, save_path=save_path, filename=cal_file_name)
            
    elif kwargs.exp_name == 'quantize':
         
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            to_add = kwargs.data + '_' + 'shift' + '_calsize_' + f'{kwargs.dataset.subsample}'
        else:
            to_add = kwargs.data + '_calsize_' + f'{kwargs.dataset.subsample}'
        total_epochs = kwargs.models.epochs
        # if kwargs.quantize:
        #     total_epochs = kwargs.models.epochs
        # else:
        #     total_epochs =  kwargs.checkpoint.epochs
        n_bins = kwargs.n_bins_calibration_metrics  
        n_bins_esse = kwargs.n_bins_esse
        gamma = kwargs.gamma             
        name = kwargs.exp_name
        if kwargs.models.S != 64:
            name += f'slot-{kwargs.models.S}'
        if kwargs.models.K != 64:
            name += f'kappa-{kwargs.models.K}'
        if kwargs.models.random:
            name += '_random'
        if kwargs.models.L1:
            name += '_L1'       
        if kwargs.models.quantization_only: 
            name += '_quantonly' 
        if kwargs.models.standard_dirichlet:
            name += '_stdcal'    
        if kwargs.models.quadratic:
            name += '_quadratic'
            
        if kwargs.data == 'synthetic':
            appendix = name + '_' + to_add + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
            test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)    
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_seed-{}_ep-{}_{}.csv".format(
                    name, #kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed, #kwargs.checkpoint.seed,
                    total_epochs,                
                    model_class
                )        
        else: 
            appendix = name + '_' + to_add + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'            
            test_file_name = 'multicalss_quantisation_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)       
            if kwargs.corruption_type:
                test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                    name, #kwargs.exp_name,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    corruption_text,
                    kwargs.seed, #kwargs.checkpoint.seed,
                    total_epochs,                
                    model_class
                )        
            else:      
                if kwargs.data == 'weather' and kwargs.dataset.shift:
                    test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_shift_seed-{}_ep-{}_{}.csv".format(
                        name, #kwargs.exp_name,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
                    )
                else:               
                    test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_calquant_seed-{}_ep-{}_{}.csv".format(
                            name, #kwargs.exp_name,
                            kwargs.data,
                            kwargs.dataset.num_classes,
                            kwargs.dataset.num_features,
                            kwargs.seed, #kwargs.checkpoint.seed,
                            total_epochs,                
                            model_class
                        )
                
        # Load your data
        df_test = pd.read_csv(test_results)        

        # Compute accuracy
        accuracy_test = (df_test['preds'] == df_test['true']).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')  
        accs = {'acc': [accuracy_test]}
        # Convert to DataFrame
        df_accs = pd.DataFrame(accs)
        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +
        # Save to CSV
        print(f"saved accuracy to {output_file}")
        df_accs.to_csv(output_file, index=False) 
                
        # === codeword usage statistics ===
        idx_np = df_test.filter(regex=r'^indices').to_numpy().ravel()
        unique, counts = np.unique(idx_np, return_counts=True)
        freq = counts / counts.sum()

        print("\n=== Codeword usage statistics ===")
        print(f"Total codewords used: {len(unique)} / {idx_np.max() + 1}")
        print(f"Min freq: {freq.min():.6f}")
        print(f"Max freq: {freq.max():.6f}")
        print(f"Mean freq (ideal): {1.0 / (idx_np.max() + 1):.6f}")

        # Entropy (max = log K)
        entropy = -np.sum(freq * np.log(freq + 1e-12))
        max_entropy = np.log(idx_np.max() + 1)
        print(f"Entropy: {entropy:.4f} / {max_entropy:.4f}")

        # Optional: print full histogram (comment out if too verbose)
        usage_df = pd.DataFrame({
            "codeword": unique,
            "count": counts,
            "frequency": freq
        }).sort_values("codeword")
        # print(usage_df)
        if kwargs.corruption_type:
            output_file = os.path.join(output_dir, f'usage_stats_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +
        else:
            output_file = os.path.join(output_dir, f'usage_stats_seed_{kwargs.seed}_{model_class}.csv') #'metric_' + appendix +  
        usage_df.to_csv(output_file, index=False)  
        # ================================
        
        # === standard deviation of learned region dependent calibration parameters ===
        alpha_test = df_test.filter(regex=r'^alpha')
        alpha_test_ = torch.tensor(alpha_test.values, dtype=torch.float32)
        
        alpha_std = torch.std(alpha_test_, dim=0)
        alpha_mean = torch.mean(alpha_test_, dim=0)
        
        if not kwargs.only_test:             
            # Extract logits and true labels
            logits_test = df_test.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
            pca_test = df_test.filter(regex=r'^pca')            
            labels_test = df_test['true']
                        
            logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
            pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
            l2_test_ = torch.tensor(df_test.filter(regex=r'^l2').values, dtype=torch.float32)
            y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)            
            
            # Convert logits to probabilities            
            probs_test = F.softmax(logits_test_, dim=1)    
            
            # Compute calibration metrics
            if kwargs.models.adabw:
                bw_test = df_test.filter(regex=r'^bandwidth')
                bw_test = torch.tensor(bw_test.values, dtype=torch.float32).squeeze() 
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce_adabw(probs_test, y_true_test_, pca_test_, bw_test, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
            else:
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_,class_freqs, n_bins, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class) #compute_multiclass_calibration_metrics_w_lce_quantv2(probs_test, y_true_test_, pca_test_, l2_test_,class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                # ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile, l2_profile = compute_multiclass_calibration_metrics_w_lce_quantv2(probs_test, y_true_test_, pca_test_, l2_test_,class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                results = {
                    "ECCE": [ecce_test],
                    "ECE": [ece_test],
                    "MCE": [mce_test],
                    "Brier": [brier_test],
                    "NLL": [nll_test],
                    "LCE": [lce_test],
                    "MLCE": [mlce_test]
                }

                # Convert to DataFrame
                df = pd.DataFrame(results)

                # Specify your directory and filename
                output_dir = join(kwargs.save_path_calibration_metrics, appendix)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                # Save to CSV
                df.to_csv(output_file, index=False)

                # ---- Save aggregated ESS profile ----
                ess_results = {
                    "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                    "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                    "avg_ess": ess_profile["avg_ess_per_bin"],
                    "count": ess_profile["count_per_bin"]
                }

                df_ess = pd.DataFrame(ess_results)

                ess_output_file = os.path.join(
                    output_dir,
                    f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
                )

                df_ess.to_csv(ess_output_file, index=False)

                # Save to CSV
                df.to_csv(output_file, index=False)
                print(f"Test Quantisation — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")
            #else:
                all_lce = []
                all_mlce = []
                for gamma in kwargs.gammas:
                    print(f'Computing metrics with gamma {gamma}')
                    ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                    all_lce.append(lce_test)
                    all_mlce.append(mlce_test)
                if gamma != kwargs.gamma:

                    gamma_plot = {
                        'GAMMA': kwargs.gammas,
                        'LCE': all_lce,
                        'MLCE': all_mlce
                    }

                    # Convert to DataFrame
                    df = pd.DataFrame(gamma_plot)

                    # Specify your directory and filename
                    output_dir = join(kwargs.save_path_calibration_metrics, appendix)
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                    # Save to CSV
                    df.to_csv(output_file, index=False)
                        
            print("probs_test min/max:", probs_test.min().item(), probs_test.max().item())
            # Calibration plot        
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy)
    elif kwargs.exp_name == 'calibrate':          
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            to_add = kwargs.data + '_' + 'shift' + '_calsize_' + f'{kwargs.dataset.subsample}'
        else:
            to_add = kwargs.data + '_calsize_' + f'{kwargs.dataset.subsample}'
        if kwargs.calibrate:
            total_epochs = kwargs.models.epochs
        else:
            total_epochs =  kwargs.checkpoint.epochs
        n_bins = kwargs.n_bins_calibration_metrics  
        n_bins_esse = kwargs.n_bins_esse
        gamma = kwargs.gamma              
        if kwargs.data == 'synthetic':
            appendix = kwargs.exp_name + '_' + to_add + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
            test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)    
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed, #kwargs.checkpoint.seed,
                    total_epochs,                
                )        
        else:  
            appendix = kwargs.exp_name + '_' + to_add + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
            if kwargs.models.lambda_kl == 0:
                appendix = 'reference_kernel' + '_' + to_add + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
            if kwargs.models.kernel_only:
                appendix = 'kernel_only' + '_' + to_add + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
            test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)    
            if kwargs.corruption_type:
                test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        corruption_text,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
                    )
                if kwargs.models.lambda_kl == 0:
                    test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        'reference_kernel',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        corruption_text,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
                    )
                if kwargs.models.kernel_only:
                    test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        'kernel_only',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        corruption_text,
                        kwargs.seed, #kwargs.checkpoint.seed,
                        total_epochs,                
                        model_class
                    )
            else:       
                root = "results/{}/{}_{}_classes_{}_features/".format(
                    kwargs.exp_name,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features
                )                          
                if kwargs.models.lambda_kl == 0:
                    root = "results/{}/{}_{}_classes_{}_features/".format(
                        'reference_kernel',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features
                    )
                if kwargs.models.kernel_only:
                    root = "results/{}/{}_{}_classes_{}_features/".format(
                        'kernel_only',
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features
                    )
                piece = f"raw_results_test_cal_seed-{kwargs.seed}_ep-{total_epochs}_{model_class}.csv"
                if kwargs.data == 'weather' and kwargs.dataset.shift:
                    piece = f"raw_results_test_cal_shift_seed-{kwargs.seed}_ep-{total_epochs}_{model_class}.csv"  
                test_results = root + piece   

                   
        # Load your data
        df_test = pd.read_csv(test_results)        

        # Compute accuracy
        accuracy_test = (df_test['preds'] == df_test['true']).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')  
        accs = {'acc': [accuracy_test]}
        # Convert to DataFrame
        df_accs = pd.DataFrame(accs)
        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +
        # Save to CSV
        df_accs.to_csv(output_file, index=False)      
        
        if not kwargs.only_test: 
            # Extract logits and true labels
            logits_test = df_test.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
            pca_test = df_test.filter(regex=r'^features')
            labels_test = df_test['true']
            
            logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
            pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
            y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)
            
            # Convert logits to probabilities
            if kwargs.models.lambda_kl == 0 or kwargs.models.kernel_only:
                probs_test = logits_test_             
            else:
                probs_test = F.softmax(logits_test_, dim=1)    
            
            # Compute calibration metrics
            if kwargs.models.adabw:
                bw_test = df_test.filter(regex=r'^bandwidth')
                bw_test = torch.tensor(bw_test.values, dtype=torch.float32).squeeze() 
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce_adabw(probs_test, y_true_test_, pca_test_, bw_test, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
            else:
                if kwargs.calibrate:
                    ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                    results = {
                        "ECCE": [ecce_test],       
                        "ECE": [ece_test],
                        "MCE": [mce_test],
                        "Brier": [brier_test],
                        "NLL": [nll_test],
                        "LCE": [lce_test],
                        "MLCE": [mlce_test]
                    }

                    # Convert to DataFrame
                    df = pd.DataFrame(results)

                    # Specify your directory and filename
                    output_dir = join(kwargs.save_path_calibration_metrics, appendix)
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                    # Save to CSV
                    df.to_csv(output_file, index=False)  
                    
                    # ---- Save aggregated ESS profile ----
                    ess_results = {
                        "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                        "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                        "avg_ess": ess_profile["avg_ess_per_bin"],
                        "count": ess_profile["count_per_bin"]
                    }

                    df_ess = pd.DataFrame(ess_results)

                    ess_output_file = os.path.join(
                        output_dir,
                        f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
                    )
                    
                    df_ess.to_csv(ess_output_file, index=False)
                    print(f"Saved ESS profile to {ess_output_file}")
                    # print(f"Not Saved ESS profile to {ess_output_file}. Uncomment the line!")
                    
                    # Print results
                    print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                else:    
                    all_lce = []
                    all_mlce = []                
                    for gamma in kwargs.gammas:
                        print(f'Computing metrics with gamma {gamma}')
                        ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                        all_lce.append(lce_test)
                        all_mlce.append(mlce_test)
                        if gamma == kwargs.gamma:
                            results = {
                                "ECCE": [ecce_test],       
                                "ECE": [ece_test],
                                "MCE": [mce_test],
                                "Brier": [brier_test],
                                "NLL": [nll_test],
                                "LCE": [lce_test],
                                "MLCE": [mlce_test]
                            }

                            # Convert to DataFrame
                            df = pd.DataFrame(results)

                            # Specify your directory and filename
                            output_dir = join(kwargs.save_path_calibration_metrics, appendix)
                            os.makedirs(output_dir, exist_ok=True)
                            output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                            # Save to CSV
                            df.to_csv(output_file, index=False)  
                            
                            # ---- Save aggregated ESS profile ----
                            ess_results = {
                                "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                                "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                                "avg_ess": ess_profile["avg_ess_per_bin"],
                                "count": ess_profile["count_per_bin"]
                            }

                            df_ess = pd.DataFrame(ess_results)

                            ess_output_file = os.path.join(
                                output_dir,
                                f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
                            )
                            
                            df_ess.to_csv(ess_output_file, index=False)
                            
                            # Print results
                            print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                    
                    gamma_plot = {
                        'GAMMA': kwargs.gammas,
                        'LCE': all_lce,
                        'MLCE': all_mlce
                    }
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(gamma_plot)

                    # Specify your directory and filename
                    output_dir = join(kwargs.save_path_calibration_metrics, appendix)
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                    # Save to CSV
                    df.to_csv(output_file, index=False)                                                             
                    
            print("probs_test min/max:", probs_test.min().item(), probs_test.max().item())
            # Calibration plot        
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy)
                
    elif kwargs.exp_name == 'competition':            
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            to_add = kwargs.data + '_' + 'shift' + '_calsize_' + f'{kwargs.dataset.subsample}'
        else:
            to_add = kwargs.data + '_calsize_' + f'{kwargs.dataset.subsample}'                 
        n_bins = kwargs.n_bins_calibration_metrics 
        n_bins_esse = kwargs.n_bins_esse
        gamma = kwargs.gamma 
                                
        appendix = kwargs.exp_name + '_' 
        appendix += kwargs.method 
        appendix += '_'+ to_add + '_' 
        appendix += f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'
        test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
        save_path = join(kwargs.save_path_calibration_plots, appendix)
        os.makedirs(save_path, exist_ok=True)  
        if kwargs.corruption_type:              
            test_results = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_corrupt_{}_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        corruption_text,
                        kwargs.seed,
                        kwargs.models.max_iter,                
                        model_class
                    )
        else:
            if kwargs.data == 'weather' and kwargs.dataset.shift:
                test_results = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_shift_seed-{}_ep-{}_{}.csv".format(
                        kwargs.exp_name,
                        kwargs.method,
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.seed,
                        kwargs.models.max_iter,                
                        model_class
                    )
            else:
                test_results = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}_{}.csv".format(
                            kwargs.exp_name,
                            kwargs.method,
                            kwargs.data,
                            kwargs.dataset.num_classes,
                            kwargs.dataset.num_features,
                            kwargs.seed,
                            kwargs.models.max_iter,                
                            model_class
                        )
        
        # Load your data
        df_test = pd.read_csv(test_results)        

        # Compute accuracy
        accuracy_test = (df_test['preds'] == df_test['true']).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')  
        accs = {'acc': [accuracy_test]}
        # Convert to DataFrame
        df_accs = pd.DataFrame(accs)
        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +
        # Save to CSV
        df_accs.to_csv(output_file, index=False)    
               
        if not kwargs.only_test: 
            # Extract logits and true labels
            logits_test = df_test.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
            pca_test = df_test.filter(regex=r'^features')
            labels_test = df_test['true']
            
            logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
            pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
            y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)

            # Convert logits to probabilities
            if kwargs.method in ['SMS', 'DC', 'IR', 'PS', 'PC']:
                probs_test = logits_test_ #F.softmax(logits_test_, dim=1)              
            else:
                probs_test = F.softmax(logits_test_, dim=1)              
                    
            # Compute calibration metrics
            if kwargs.models.adabw:
                bw_data = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                        'calibrate',                    
                        kwargs.data,
                        kwargs.dataset.num_classes,
                        kwargs.dataset.num_features,
                        kwargs.seed,
                        kwargs.checkpoint.epochs_bw,                
                    )        
                # Load your data
                df_bw = pd.read_csv(bw_data)               
                bw_test = df_bw.filter(regex=r'^bandwidth')
                bw_test = torch.tensor(bw_test.values, dtype=torch.float32).squeeze() 
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce_adabw(probs_test, y_true_test_, pca_test_, bw_test, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
            else:                               
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
            
                results = {
                    "ECCE": [ecce_test],       
                    "ECE": [ece_test],
                    "MCE": [mce_test],
                    "Brier": [brier_test],
                    "NLL": [nll_test],
                    "LCE": [lce_test],
                    "MLCE": [mlce_test]
                }

                # Convert to DataFrame
                df = pd.DataFrame(results)

                # Specify your directory and filename
                output_dir = join(kwargs.save_path_calibration_metrics, appendix)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                # Save to CSV
                df.to_csv(output_file, index=False)   
                
                # ---- Save aggregated ESS profile ----
                ess_results = {
                    "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                    "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                    "avg_ess": ess_profile["avg_ess_per_bin"],
                    "count": ess_profile["count_per_bin"]
                }

                df_ess = pd.DataFrame(ess_results)

                ess_output_file = os.path.join(
                    output_dir,
                    f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
                )
                
                df_ess.to_csv(ess_output_file, index=False)
                
                # Print results
                print(f"Test Calibration — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                    
                all_lce = []
                all_mlce = []
                for gamma in kwargs.gammas:
                    print(f'Computing metrics with gamma {gamma}')     
                    ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                    all_lce.append(lce_test)
                    all_mlce.append(mlce_test)

                    if gamma != kwargs.gamma:
                        
                        gamma_plot = {
                            'GAMMA': kwargs.gammas,
                            'LCE': all_lce,
                            'MLCE': all_mlce
                        }
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(gamma_plot)

                        # Specify your directory and filename
                        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_corrupt_{corruption_text}_seed_{kwargs.seed}_{model_class}.csv') #'metric_' + appendix +

                        # Save to CSV
                        df.to_csv(output_file, index=False)                                   
            
            # Calibration plot            
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy)                
            
    elif kwargs.exp_name == 'replicate':
        
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            to_add = kwargs.data + '_' + 'shift' + '_calsize_' + f'{kwargs.dataset.subsample}'
        else:
            to_add = kwargs.data + '_calsize_' + f'{kwargs.dataset.subsample}'
        total_epochs = kwargs.models.max_iter   
        n_bins = kwargs.n_bins_calibration_metrics  
        n_bins_esse = kwargs.n_bins_esse
        gamma = kwargs.gamma             
        name = kwargs.exp_name
        name += f'{kwargs.models.n_steps}'
        if kwargs.models.kl_reg > 0:
            name += '_KL'
        if kwargs.models.state_dependent:
            name += '_DEP'
            
        if kwargs.data == 'synthetic':
            appendix = name + '_' + kwargs.data + '_' + f'{kwargs.checkpoint.num_classes}_classes_' + f'{kwargs.checkpoint.num_features}_features'
            test_file_name = 'multicalss_calibration_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)    
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_replicate_seed-{}_ep-{}.csv".format(
                    name, 
                    kwargs.data,
                    kwargs.checkpoint.num_classes,
                    kwargs.checkpoint.num_features,
                    kwargs.seed, #kwargs.checkpoint.seed,
                    total_epochs,                
                )        
        else: 
            appendix = name + '_' + to_add + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'            
            test_file_name = 'multicalss_replicate_test_' + f'{kwargs.bin_strategy}' + '.png'        
            save_path = join(kwargs.save_path_calibration_plots, appendix)
            os.makedirs(save_path, exist_ok=True)                            
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_test_replicate_seed-{}_ep-{}_{}.csv".format(
                    name, 
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.seed,
                    kwargs.models.max_iter,
                    model_class           
                )
            
        # Load your data
        df_test = pd.read_csv(test_results)        

        # Compute accuracy
        accuracy_test = (df_test['preds'] == df_test['true']).mean()
        print(f'Test accuracy: {accuracy_test:.2%}')  
        accs = {'acc': [accuracy_test]}
        # Convert to DataFrame
        df_accs = pd.DataFrame(accs)
        # Specify your directory and filename
        output_dir = join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'accs_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +
        # Save to CSV
        df_accs.to_csv(output_file, index=False) 
                
        if not kwargs.only_test: # if only_test only computes accuracy            
            # Extract logits and true labels
            logits_test = df_test.filter(regex=r'^logits') #df_test.drop(columns=['preds', 'true'])
            pca_test = df_test.filter(regex=r'^features')
            labels_test = df_test['true']
                        
            logits_test_ = torch.tensor(logits_test.values, dtype=torch.float32)
            pca_test_ = torch.tensor(pca_test.values, dtype=torch.float32)
            y_true_test_ = torch.tensor(labels_test.values, dtype=torch.long)            
            
            # Convert logits to probabilities            
            probs_test = logits_test_ #F.softmax(logits_test_, dim=1)    
            
            # Compute calibration metrics
            if kwargs.models.adabw:
                bw_test = df_test.filter(regex=r'^bandwidth')
                bw_test = torch.tensor(bw_test.values, dtype=torch.float32).squeeze() 
                ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test = compute_multiclass_calibration_metrics_w_lce_adabw(probs_test, y_true_test_, pca_test_, bw_test, n_bins, gamma=gamma, bin_strategy=kwargs.bin_strategy) 
            else:
                if kwargs.replicate or kwargs.test:
                    if kwargs.data == 'cubic':
                        ecce_test, ece_test, mce_test, brier_test, nll_test = compute_multiclass_calibration_metrics(probs_test, y_true_test_, n_bins, class_freqs) 
                        results = {
                            "ECCE": [ecce_test],       
                            "ECE": [ece_test],
                            "MCE": [mce_test],
                            "Brier": [brier_test],
                            "NLL": [nll_test]                            
                        }
                    else:
                        ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=kwargs.gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                        results = {
                            "ECCE": [ecce_test],       
                            "ECE": [ece_test],
                            "MCE": [mce_test],
                            "Brier": [brier_test],
                            "NLL": [nll_test],
                            "LCE": [lce_test],
                            "MLCE": [mlce_test]
                        }

                    # Convert to DataFrame
                    df = pd.DataFrame(results)

                    # Specify your directory and filename
                    output_dir = join(kwargs.save_path_calibration_metrics, appendix)
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f'metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                    # Save to CSV
                    df.to_csv(output_file, index=False)  
                    
                    # ---- Save aggregated ESS profile ----
                    ess_results = {
                        "ess_bin": list(range(len(ess_profile["avg_abs_lce_per_ess_bin"]))),
                        "avg_abs_lce": ess_profile["avg_abs_lce_per_ess_bin"],
                        "avg_ess": ess_profile["avg_ess_per_bin"],
                        "count": ess_profile["count_per_bin"]
                    }

                    df_ess = pd.DataFrame(ess_results)

                    ess_output_file = os.path.join(
                        output_dir,
                        f"ess_profile_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv"
                    )
                    
                    df_ess.to_csv(ess_output_file, index=False)
                    
                    # Print results
                    if kwargs.data == 'cubic':
                        print(f"Test Replicator — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}") #, LCE: {lce_test:.4f}, MLCE: {mlce_test:.4f}")        
                    else:
                        print(f"Test Replicator — ECCE: {ecce_test:.4f}, ECE: {ece_test:.4f}, MCE: {mce_test:.4f}, Brier: {brier_test:.4f}, NLL: {nll_test:.4f}, LCE: {lce_test:.4f}") #, MLCE: {mlce_test:.4f}")        
                 
                    all_lce = []
                    all_mlce = []       
                    if kwargs.data != 'cubic':         
                        for gamma in kwargs.gammas:
                            print(f'Computing metrics with gamma {gamma}')
                            ecce_test, ece_test, mce_test, brier_test, nll_test, lce_test, mlce_test, ess_profile = compute_multiclass_calibration_metrics_w_lce(probs_test, y_true_test_, pca_test_, class_freqs, n_bins, n_bins_esse, gamma=gamma, bin_strategy=kwargs.bin_strategy, data=kwargs.data, model_type=model_class)
                            all_lce.append(lce_test)
                            all_mlce.append(mlce_test)

                        if gamma != kwargs.gamma:
                            
                            gamma_plot = {
                                'GAMMA': kwargs.gammas,
                                'LCE': all_lce,
                                'MLCE': all_mlce
                            }
                            
                            # Convert to DataFrame
                            df = pd.DataFrame(gamma_plot)

                            # Specify your directory and filename
                            output_dir = join(kwargs.save_path_calibration_metrics, appendix)
                            os.makedirs(output_dir, exist_ok=True)
                            output_file = os.path.join(output_dir, f'gamma_plot_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{kwargs.seed}_corrupt_{corruption_text}_{model_class}.csv') #'metric_' + appendix +

                            # Save to CSV
                            df.to_csv(output_file, index=False)                                                             
                        
            print("probs_test min/max:", probs_test.min().item(), probs_test.max().item())
            # Calibration plot        
            multiclass_calibration_plot(y_true_test_, probs_test, n_bins=n_bins, save_path=save_path, filename=test_file_name, bin_strategy=kwargs.bin_strategy) 
    
    elif kwargs.exp_name == 'ess_plot': 
        data_names = ['cifar10', 'cifar100', 'tissue'] #, 'weather']
        agg_dicts = []
        for data in data_names:
            if kwargs.l2_plot:
                save_path = "results/metrics_vit"  # change if needed
                summary = plot_entropy_lce(args=kwargs, save_path=save_path)         
                save_pipe_table(summary, save_dir=save_path, filename=f"entropy_lce_table_{kwargs.data}.txt")
            else:
                if data == 'cifar100':                
                    data_name = 'CIFAR-100'
                    # model_in_title = 'ResNet52'
                elif data == 'cifar10':                
                    data_name = 'CIFAR-10'
                    # model_in_title = 'ResNet152'
                elif data == 'tissue':                
                    data_name = 'TissueMNIST'
                    # model_in_title = 'ResNet52'
                elif data == 'weather':
                    data_name = 'Weather'
                    # model_in_title = 'FTT'

                model_class = 'vit'
                metrics_root = "results/metrics_vit"  # change if needed

                method_to_runs = collect_ess_profiles(metrics_root, data, model_class)
                agg_dict = aggregate_method_runs_2(method_to_runs)
                agg_dicts.append(agg_dict)

                print("Methods found:")
                for method, runs in method_to_runs.items():
                    print(f"  {method}: {len(runs)} runs")

        # plot_ess_profiles_2(
        #     agg_dict,
        #     save_path=os.path.join(metrics_root, f"ess_profile_comparison_{kwargs.data}_{model_class}.pdf"),
        #     # title="", #"Average Absolute LCE Across Density Bins for ", # + f"{data_name} with a " + f"{model_in_title}", # ,
        #     interval="std",   # use "sem95" for 95% confidence band,
        #     data_name=kwargs.data,
        # )
        
        plot_ess_profiles_together(
            agg_dicts,
            save_path=os.path.join(metrics_root, f"ess_profile_comparison_together_{model_class}.pdf"),
            # title="", #"Average Absolute LCE Across Density Bins for ", # + f"{data_name} with a " + f"{model_in_title}", # ,
            interval="std",   # use "sem95" for 95% confidence band,
            data_names=data_names,
        )

    elif kwargs.exp_name == 'ablate_cal_size':
        method_names = ['VQ', 'LCN', 'RK', 'SMS', 'DC', 'PS', 'TS', 'IR', 'PC']
        data_names = ['weather', 'tissue']
        for data_name in data_names:
            for method_name in method_names: 
                # method_name = "RK" # change if needed
                # data_name = "tissue" # change if needed
                
                if method_name == "VQ":
                    method = "quantize"  
                elif method_name == "LCN":
                    method = "calibrate"
                elif method_name == "RK":
                    method = "reference_kernel"
                elif method_name == "SMS":
                    method = "competition_SMS"
                elif method_name == "DC":
                    method = "competition_DC"
                elif method_name == "PS":
                    method = "competition_PS"
                elif method_name == "TS":
                    method = "competition_TS"
                elif method_name == "IR":
                    method = "competition_IR"   
                elif method_name == "PC":
                    method = "competition_PC" 
                    
                if data_name == "tissue":
                    num_classes = 8
                elif data_name == "weather":
                    num_classes = 5
                
                calsizes = (0.05, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0)
                seeds = range(42, 47)
                
                result_table = summarize_vq_by_calsize(base_dir=kwargs.save_path_calibration_metrics,
                    calsizes=calsizes,
                    seeds=seeds,
                    method=method,
                    data_name=data_name,
                    num_classes=num_classes,        
                    method_name=method_name,                    
                )                                                    
                
                output_dir = os.path.join(kwargs.save_path_calibration_metrics, "ablate_cal_size_results_w_acc")
                os.makedirs(output_dir, exist_ok=True)
                print(output_dir)
                result_table.to_csv(os.path.join(output_dir, f"ablate_{method_name}_{data_name}.csv"), index=False)     
                print(f"\nSaved ablation results for {method_name} on {data_name} to {output_dir}")
                
                if data_name == "weather":
                    result_table = summarize_vq_by_calsize(base_dir=kwargs.save_path_calibration_metrics,
                    calsizes=calsizes,
                    seeds=seeds,
                    method=method,
                    data_name=data_name+"_shift",
                    num_classes=num_classes,        
                    method_name=method_name,                    
                    )                                                    
                    
                    output_dir = os.path.join(kwargs.save_path_calibration_metrics, "ablate_cal_size_results_w_acc")
                    os.makedirs(output_dir, exist_ok=True)
                    print(output_dir)
                    result_table.to_csv(os.path.join(output_dir, f"ablate_{method_name}_{data_name}_shift.csv"), index=False)     
                    print(f"\nSaved ablation results for {method_name} on {data_name} to {output_dir}")
                    
    elif kwargs.exp_name == 'ablate_s_k_tissue':
        slots=(16, 32, 64, 128, 256)
        kappas=(16, 32, 64, 128, 256)
        seeds=range(42, 47)
        result_slots, result_kappa = summarize_vq_by_slot_kappa(
            base_dir=kwargs.save_path_calibration_metrics,
            slots=slots,
            kappas=kappas,
            seeds=seeds,            
            method_name="VQ"
        )
        output_dir = os.path.join(kwargs.save_path_calibration_metrics, "ablate_s_k_tissue_results_w_acc_fixed")
        os.makedirs(output_dir, exist_ok=True)
        print(output_dir)
        result_slots.to_csv(os.path.join(output_dir, f"ablate_slots_tissue.csv"), index=False)
        result_kappa.to_csv(os.path.join(output_dir, f"ablate_kappas_tissue.csv"), index=False)
        print(f"\nSaved ablation of S and K results for tissue to {output_dir}")
        
    elif kwargs.exp_name == 'convnext_results':
        method_names = ['VQ', 'LCN', 'RK', 'SMS', 'DC', 'PS', 'TS', 'IR', 'PC', 'NC']
        data_names = ['cifar10', 'cifar100', 'tissue'] #, 'weather', 'weather_shift'] # ['tissue'] 
        model_class = 'vit' #'resnet' #'convnext' #'resnet'
        dir = 

        for data_name in data_names:
            reusults_tables = []
            for method_name in method_names: 
                # method_name = "RK" # change if needed
                # data_name = "tissue" # change if needed
                
                if method_name == "VQ":
                    method = "quantize"  
                elif method_name == "LCN":
                    method = "calibrate"
                elif method_name == "RK":
                    method = "reference_kernel"
                elif method_name == "SMS":
                    method = "competition_SMS"
                elif method_name == "DC":
                    method = "competition_DC"
                elif method_name == "PS":
                    method = "competition_PS"
                elif method_name == "TS":
                    method = "competition_TS"
                elif method_name == "IR":
                    method = "competition_IR"   
                elif method_name == "PC":
                    method = "competition_PC" 
                elif method_name == "NC":
                    method = "pre-train"
                    
                if data_name == "tissue":
                    num_classes = 8
                elif data_name in ["weather", "weather_shift"]:
                    num_classes = 5
                elif data_name == "cifar10":
                    num_classes = 10
                elif data_name == "cifar100":
                    num_classes = 100
                
                calsizes = [1.0]
                seeds = range(42, 47)
                
                result_table = summarize_vq_by_calsize(base_dir=dir,
                    calsizes=calsizes,
                    seeds=seeds,
                    method=method,
                    data_name=data_name,
                    num_classes=num_classes,        
                    method_name=method_name,   
                    model_class=model_class                 
                )                                                    
                
                reusults_tables.append(result_table)
        
            final_table = pd.concat(reusults_tables, ignore_index=True)   
            final_table.drop(columns=['calsize'], inplace=True)
                
            output_dir = os.path.join(dir, f"summary_{model_class}_results_w_acc")
            os.makedirs(output_dir, exist_ok=True) 
                
            final_table.to_csv(os.path.join(output_dir, f"{data_name}.csv"), index=False)     
            print(f"\nSaved ablation results for {method_name} on {data_name} to {output_dir}")
    
    elif kwargs.exp_name == 'ablate_vq':
        method_names = ['VQ', 'VQ-DC', 'VQ-NC', 'NC']
        data_names = ['tissue', 'weather']
                
        for data_name in data_names:
            reusults_tables = []
            for method_name in method_names: 
                # method_name = "RK" # change if needed
                # data_name = "tissue" # change if needed
                
                if method_name == "VQ":
                    method = "quantize"  
                if method_name == "VQ-DC":
                    method = "quantize_stdcal" 
                if method_name == "VQ-NC":
                    method = "quantize_quantonly" 
                elif method_name == "LCN":
                    method = "calibrate"
                elif method_name == "RK":
                    method = "reference_kernel"
                elif method_name == "SMS":
                    method = "competition_SMS"
                elif method_name == "DC":
                    method = "competition_DC"
                elif method_name == "PS":
                    method = "competition_PS"
                elif method_name == "TS":
                    method = "competition_TS"
                elif method_name == "IR":
                    method = "competition_IR"   
                elif method_name == "PC":
                    method = "competition_PC" 
                elif method_name == "NC":
                    method = "pre-train"
                    
                if data_name == "tissue":
                    num_classes = 8
                elif data_name == "weather":
                    num_classes = 5
                elif data_name == "cifar10":
                    num_classes = 10
                elif data_name == "cifar100":
                    num_classes = 100
                
                calsizes = [1.0]
                seeds = range(42, 47)
                
                result_table = summarize_vq_by_calsize(base_dir=kwargs.save_path_calibration_metrics,
                    calsizes=calsizes,
                    seeds=seeds,
                    method=method,
                    data_name=data_name,
                    num_classes=num_classes,        
                    method_name=method_name                 
                )                                                    
                
                reusults_tables.append(result_table)
        
            final_table = pd.concat(reusults_tables, ignore_index=True)   
            final_table.drop(columns=['calsize'], inplace=True)
                
            output_dir = os.path.join(kwargs.save_path_calibration_metrics, "ablate_VQ_w_acc")
            os.makedirs(output_dir, exist_ok=True) 
                
            final_table.to_csv(os.path.join(output_dir, f"{data_name}.csv"), index=False)     
            print(f"\nSaved ablation results for {method_name} on {data_name} to {output_dir}")


    elif kwargs.exp_name == 'corruption_results':
        method_names = ['VQ', 'LCN', 'RK', 'SMS', 'DC', 'PS', 'TS', 'IR', 'PC', 'NC']
        data_name = 'cifar10' #['cifar10', 'cifar100', 'tissue']
        corruptions = ['brightness', 'pixelate'] #'frost'
        model_class = 'resnet' #'convnext'
        severities = range(1,6)

        for corruption in corruptions:
            for severity in severities: 
                reusults_tables = []
                for method_name in method_names: 
                    # method_name = "RK" # change if needed
                    # data_name = "tissue" # change if needed
                    
                    if method_name == "VQ":
                        method = "quantize"  
                    elif method_name == "LCN":
                        method = "calibrate"
                    elif method_name == "RK":
                        method = "reference_kernel"
                    elif method_name == "SMS":
                        method = "competition_SMS"
                    elif method_name == "DC":
                        method = "competition_DC"
                    elif method_name == "PS":
                        method = "competition_PS"
                    elif method_name == "TS":
                        method = "competition_TS"
                    elif method_name == "IR":
                        method = "competition_IR"   
                    elif method_name == "PC":
                        method = "competition_PC" 
                    elif method_name == "NC":
                        method = "pre-train"
                        
                    if data_name == "tissue":
                        num_classes = 8
                    elif data_name == "weather":
                        num_classes = 5
                    elif data_name == "cifar10":
                        num_classes = 10
                    elif data_name == "cifar100":
                        num_classes = 100
                    
                    calsizes = [1.0]
                    seeds = range(42, 47)
                    base_dir = 'results/metrics_shifts/'
                    result_table = summarize_vq_by_calsize(base_dir=base_dir,
                        calsizes=calsizes,
                        seeds=seeds,
                        method=method,
                        data_name=data_name,
                        num_classes=num_classes,        
                        method_name=method_name,   
                        model_class=model_class,
                        corruption=corruption+f'_severity_{severity}'                 
                    )                                                    
                    
                    reusults_tables.append(result_table)
            
                final_table = pd.concat(reusults_tables, ignore_index=True)   
                final_table.drop(columns=['calsize'], inplace=True)
                    
                output_dir = os.path.join(base_dir, f"summary_{model_class}_results_w_acc")
                os.makedirs(output_dir, exist_ok=True) 
                    
                final_table.to_csv(os.path.join(output_dir, f"{data_name}_{corruption}_severity_{severity}.csv"), index=False)     
                print(f"\nSaved ablation results for {method_name} on {data_name} to {output_dir}")
    
        
    elif kwargs.exp_name == 'calsize_plots':
        
        base_dir = kwargs.save_path_calibration_metrics
        folder = os.path.join(base_dir, "ablate_cal_size_results_w_acc")

        methods = ['VQ', 'LCN', 'RK', 'SMS', 'DC', 'PS', 'TS', 'IR', 'PC']
        
        # datasets = ['tissue', 'weather', 'weather_shift']

        metrics = ['ECCE', 'ECE', 'NLL', 'LCE', 'MLCE', 'acc']
        datasets = ['tissue', 'weather']   # no weather_shift
        plot_metrics = ['LCE', 'NLL']      # left, right
        
        dict_marker = {
            "LCN": "d",
            "SMS": "h",
            "DC": "s",
            "IR": "H",
            "VQ": "o",
            "TS": "v",
            "PS": "P",            
            "RK": "*",
            "PC": "D",
        }
        
        legend_names = {
            "LCN": r"\textsc{LN}",
            "SMS": r"\textsc{SM}",
            "DC": r"\textsc{DC}",
            "IR": r"\textsc{IR}",
            "VQ": r"\textsc{VQ}",
            "TS": r"\textsc{TS}",
            "PS": r"\textsc{PS}",            
            "RK": r"\textsc{KC}",
            "PC": r"\textsc{PC}",
        }
        
        tick_map = {
            0.05: r"$\approx 2$",
            0.1:  r"$\approx 4$",
            0.25: r"$\approx 10$",
            0.4:  r"$\approx 15$",
            0.5:  r"$\approx 20$",
            0.75: r"$\approx 30$",
            1.0:  r"$\approx 40$",
        }
        output_folder = os.path.join(base_dir, "ablate_cal_size_results_w_acc") #"calsize_plots"
        # os.makedirs(output_folder, exist_ok=True)
        
        plt.rcParams.update({
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "font.size": 12
        })

        sns.set_theme(
            style="whitegrid",
            font_scale=1.4,
            rc={
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsfonts}\usepackage{amsmath}\usepackage{bm}",
                "font.family": "serif",
            }
        )
        
        # sns.set_palette("colorblind")

        # sns.set_theme(style="whitegrid", context="paper")
        palette = sns.color_palette("colorblind", n_colors=len(methods))

        def extract_mean_std(value):
            """
            Converts '0.006813 +- 0.000572' -> (mean, std)
            """
            if pd.isna(value):
                return None, None
            if isinstance(value, str):
                parts = value.split("+-")
                mean = float(parts[0].strip())
                std = float(parts[1].strip())
                return mean, std
            return float(value), 0.0


        # for data in datasets:
        #     all_dfs = []

        #     for method in methods:
        #         file_path = os.path.join(folder, f"ablate_{method}_{data}.csv")

        #         if not os.path.exists(file_path):
        #             print(f"Missing file: {file_path}")
        #             continue

        #         df = pd.read_csv(file_path)

        #         # Extract mean + std for each metric
        #         for metric in metrics:
        #             means, stds = zip(*df[metric].apply(extract_mean_std))
        #             df[metric] = means
        #             df[f"{metric}_std"] = stds

        #         df["method"] = method
        #         all_dfs.append(df)

        #     if not all_dfs:
        #         continue

        #     data_df = pd.concat(all_dfs, ignore_index=True)
        #     # for metric in metrics:
        #     #     print(data_df[[metric, f"{metric}_std"]].head())

        #     # ---- plotting ----
        #     for metric in metrics:
        #         plt.figure(figsize=(7, 5))

        #         for i, method in enumerate(methods):
        #             subset = data_df[data_df["method"] == method]

        #             color = palette[i]

        #             # Line
        #             plt.plot(
        #                 subset["calsize"],
        #                 subset[metric],
        #                 marker={m: dict_marker.get(m, "o") for m in methods}[method],
        #                 linewidth=2,
        #                 label=legend_names.get(method, method),
        #                 color=color
        #             )

        #             # Shaded std band
        #             plt.fill_between(
        #                 subset["calsize"],
        #                 subset[metric] - subset[f"{metric}_std"],
        #                 subset[metric] + subset[f"{metric}_std"],
        #                 color=color,
        #                 alpha=0.2
        #             )

        #         plt.xlabel(r"Calibration set size $\times 10^3$")
        #         plt.ylabel(metric)
        #         # plt.title(f"{metric} vs calibration set size - {data}")
        #         plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
                
        #         # Apply ticks
        #         ticks = list(tick_map.keys())
        #         labels = list(tick_map.values())

        #         plt.xticks(ticks=ticks, labels=labels)
                
        #         plt.tight_layout()

        #         output_path = os.path.join(output_folder, f"plot_calsize_{metric}_{data}.pdf")
        #         plt.savefig(output_path, format="pdf", bbox_inches="tight")
        #         plt.close()

        #         print(f"Saved: {output_path}")
        
        
        for data in datasets:
            all_dfs = []

            for method in methods:
                file_path = os.path.join(folder, f"ablate_{method}_{data}.csv")

                if not os.path.exists(file_path):
                    print(f"Missing file: {file_path}")
                    continue

                df = pd.read_csv(file_path)

                for metric in metrics:
                    means, stds = zip(*df[metric].apply(extract_mean_std))
                    df[metric] = means
                    df[f"{metric}_std"] = stds

                df["method"] = method
                all_dfs.append(df)

            if not all_dfs:
                continue

            data_df = pd.concat(all_dfs, ignore_index=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

            legend_handles = []
            legend_labels = []

            for ax, metric in zip(axes, plot_metrics):

                for i, method in enumerate(methods):
                    subset = (
                        data_df[data_df["method"] == method]
                        .sort_values("calsize")
                        .copy()
                    )

                    if subset.empty:
                        continue

                    x = subset["calsize"].astype(float).to_numpy()
                    y = subset[metric].astype(float).to_numpy()
                    y_std = subset[f"{metric}_std"].astype(float).to_numpy()

                    color = palette[i]
                    marker = dict_marker.get(method, "o")

                    line = ax.plot(
                        x,
                        y,
                        marker=marker,
                        linewidth=2,
                        markersize=5,
                        label=legend_names.get(method, method),
                        color=color
                    )[0]

                    ax.fill_between(
                        x,
                        y - y_std,
                        y + y_std,
                        color=color,
                        alpha=0.2,
                        linewidth=0
                    )

                    if metric == plot_metrics[0]:
                        legend_handles.append(line)
                        legend_labels.append(legend_names.get(method, method))

                ax.set_xlabel(r"Calibration set size ($\times 10^3$)")
                ax.set_ylabel(metric)

                ax.set_xticks(list(tick_map.keys()))
                ax.set_xticklabels(list(tick_map.values()), fontsize=10)

            # One shared legend
            fig.legend(
                legend_handles,
                legend_labels,
                # title="",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1), #(0.5, 0.025),
                ncol=len(methods),
                frameon=True
            )

            plt.tight_layout()

            output_path = os.path.join(output_folder, f"plot_calsize_LCE_NLL_{data}.pdf")
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close()

            print(f"Saved: {output_path}")
            
    elif kwargs.exp_name == 'box_plots': 
        output_path = os.path.join('REPLACE', "box_plots/")       
        os.makedirs(output_path, exist_ok=True)
        
        base_folder = 'REPLACE'
        methods = ['VQ', 'LN', 'KC', 'SMS', 'DC', 'PS', 'TS', 'IR', 'PC', 'NC']
        seeds = [42, 43, 44, 45, 46]
        datasets = ['cifar10', 'cifar100', 'tissue']
        df = get_df_boxplots(base_folder, methods, seeds, datasets)

        metrics = ["LCE", "MLCE"]

        plt.rcParams.update({
                'figure.dpi': 300,  # high resolution
                'savefig.dpi': 300,
                'axes.titlesize': 20,  # title font size
                'axes.labelsize': 16,  # x/y label font size
                'xtick.labelsize': 18,  # tick label sizes
                'ytick.labelsize': 16,
                'legend.fontsize': 14,
                'font.size': 14
            })
        
        sns.set_theme(
            style="whitegrid",
            font_scale=1.4,
            rc={
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsfonts}\usepackage{amsmath}\usepackage{bm}",
                "font.family": "serif",
            }
        )
        
        palette = sns.color_palette("colorblind", n_colors=len(methods))

        long = df.melt(
            id_vars=["method", "data"],
            value_vars=metrics,
            var_name="metric",
            value_name="value"
        )

        g = sns.catplot(
            data=long[~long.method.isin(['VQ-L1', 'VQ-DC', 'VQ-NC'])],
            x="method",
            y="value",
            hue="method",
            kind="box",
            col="data",
            row="metric",
            sharey=False,
            height=2.2,
            aspect=2,
            order=['VQ', 'LN', 'KC', 'DC', 'SM', 'TS', 'IR', 'PS', 'PC', "NC"],
            hue_order=['VQ', 'LN', 'KC', 'DC', 'SM', 'TS', 'IR', 'PS', 'PC', "NC"],
            palette = palette
        )

        g.set_titles(col_template = r"$\underline{{\texttt{{col_name}}}}$", row_template="")
        n_rows = g.axes.shape[0]

        for ax in g.axes.flatten():
            ax.set_title("")

        # for j, col_name in enumerate(g.col_names):
        #     ax = g.axes[0, j]
        #     ax.set_title(col_name, pad=16, fontfamily="monospace")
        #     #title.set_underline(True)
            
        for j, col_name in enumerate(g.col_names):
            ax = g.axes[0, j]
            # ax.set_title(
            #     r"$\underline{\texttt{" + col_name.replace("_", r"\_") + r"}$",
            #     pad=16
            # )
            ax.set_title(r'\underline{\texttt{'+f'{col_name}'+'}}', pad=16)

        for i, metric in enumerate(g.row_names):
            ax = g.axes[i, 0]          # first column of each row
            ax.set_ylabel(metric)

        # Reduce spacing between facets
        g.fig.subplots_adjust(
            left=0.05,
            right=0.98,
            bottom=0.15,
            top=0.95,
            wspace=0.15,
            hspace=0.25
        )

        # Remove x-axis labels and add y-grid to all subplots
        for j, ax in enumerate(g.axes.flatten()):
            ax.set_xlabel("")          # remove x-axis label
            ax.grid(axis="y", alpha=0.5)  # add y-axis grid
            ax.margins(x=0.01)

            fmt = mticker.ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_powerlimits((0, 0))   # always scientific
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.get_offset_text().set_visible(True)
            if j == 0:
                        ax.set_ylabel("$LCE$")
            elif j == 3:
                        ax.set_ylabel("$MLCE$")

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)

        #sns.despine(trim=True)

        g.fig.canvas.draw()                 # important: compute ticklabel sizes
        g.fig.align_ylabels(g.axes[:, 0])

        plt.savefig(output_path+"box_plots.pdf", format="pdf", bbox_inches='tight', dpi=300)
        plt.close()
        
    elif kwargs.exp_name == 'usage_stats': 
        
        data_names = ['cifar10', 'cifar100', 'tissue', 'weather']
        model_classes = ['resnet', 'convnext', 'vit', 'ftt']
        seeds = [42, 43, 44, 45, 46]
        classes_dict = {
            'cifar10': 10,
            'cifar100': 100,
            'tissue': 8,
            'weather': 5
        }
        
        save_path = kwargs.save_path_calibration_metrics        
        
        dfs = []
        for data_name in data_names:
            num_classes = classes_dict[data_name]
            for model_class in model_classes:    
                
                if (data_name in ['cifar10', 'cifar100', 'tissue']) and (model_class == 'ftt'):
                    continue        
                
                elif (data_name == 'weather') and (model_class in ['resnet', 'convnext', 'vit']):
                    continue
                
                else:                
                    if data_name in ['cifar10', 'cifar100']:
                        if model_class == 'resnet':
                            folder = f"quantize_{data_name}_{num_classes}_classes_None_features"
                            base_dir = 'REPLACE'
                        elif model_class in ['convnext', 'ftt']:
                            folder = f"quantize_{data_name}_calsize_1.0_{num_classes}_classes_None_features"
                            base_dir = 'REPLACE'
                        else:
                            folder = f"quantize_{data_name}_calsize_1.0_{num_classes}_classes_None_features"
                            base_dir = 'REPLACE'
                    elif data_name in ['tissue', 'weather']:
                        if model_class == 'vit':
                            folder = f"quantize_{data_name}_calsize_1.0_{num_classes}_classes_None_features"
                            base_dir = 'REPLACE'                
                        else:
                            folder = f"quantize_{data_name}_calsize_1.0_{num_classes}_classes_None_features"
                            base_dir = 'REPLACE'                                

                    df = summarize_usage_stats(
                        base_dir,
                        folder,
                        seeds,
                        dataset_name=data_name,
                        model_class=model_class
                    )
                    dfs.append(df)
                
        agg_df = pd.concat(dfs, ignore_index=True)
                
        if save_path is not None:
            agg_df.to_csv(save_path+'usage_stats.csv', index=False)

        
