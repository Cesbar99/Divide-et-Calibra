import os      
import pandas as pd
import numpy as np
from utils.utils import *

def viz_and_test(kwargs):

    all_metrics = []  # store metric values for all seeds
    collected_results = {}  # dict: method -> {metric_name: [values across seeds]}
    method_name = "JSD"
    
    collected_results[method_name] = {}

    for seed in kwargs.seeds:     
        appendix = f"calibrate_{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features"
        
        # Directory + filename
        dir = os.path.join(kwargs.save_path_calibration_metrics, appendix)
        os.makedirs(dir, exist_ok=True)
        metrics_file = os.path.join(
            dir, 
            f"metrics_None_adabw_{kwargs.models.adabw}_seed_{seed}.csv" #{kwargs.bin_strategy}
        )
        output_file = os.path.join(
            dir, 
            f"mean_std_metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}.csv"
        )

        # Read CSV
        df = pd.read_csv(metrics_file)

        # Assumption: first row = metric names, second row = values
        metric_values = df.iloc[0].values.astype(float) if df.shape[0] == 1 else df.iloc[1].values.astype(float)
        
        # Store per metric
        for name, val in zip(df.columns.tolist(), metric_values):
            collected_results[method_name].setdefault(name, []).append(val)
        all_metrics.append(metric_values)

    # Convert list to array: shape (n_seeds, n_metrics)
    all_metrics = np.array(all_metrics)

    # Compute mean & std
    mean_metrics = np.mean(all_metrics, axis=0)
    std_metrics = np.std(all_metrics, axis=0)

    # Recreate DataFrame with metric names
    metric_names = df.columns.tolist()
    results_df = pd.DataFrame({
        "metric": metric_names,
        "mean": mean_metrics,
        "std": std_metrics
    })

    # Save to CSV
    results_df.to_csv(output_file, index=False) 
    
    # Now do the same for competitors
    for method in kwargs.methods:
        all_metrics = []  # store metric values for all seeds
        collected_results[method] = {}
        for seed in kwargs.seeds:     
            appendix = f"competition_{method}_{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features"
            
            # Directory + filename
            dir = os.path.join(kwargs.save_path_calibration_metrics, appendix)
            os.makedirs(dir, exist_ok=True)
            metrics_file = os.path.join(
                dir, 
                f"metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}_seed_{seed}.csv"
            )
            output_file = os.path.join(
                dir, 
                f"mean_std_metrics_{kwargs.bin_strategy}_adabw_{kwargs.models.adabw}.csv"
            )

            # Read CSV
            df = pd.read_csv(metrics_file)

            # Assumption: first row = metric names, second row = values
            metric_values = df.iloc[0].values.astype(float) if df.shape[0] == 1 else df.iloc[1].values.astype(float)

            for name, val in zip(df.columns.tolist(), metric_values):
                collected_results[method].setdefault(name, []).append(val)
                
            all_metrics.append(metric_values)            

        # Convert list to array: shape (n_seeds, n_metrics)
        all_metrics = np.array(all_metrics)

        # Compute mean & std
        mean_metrics = np.mean(all_metrics, axis=0)
        std_metrics = np.std(all_metrics, axis=0)

        # Recreate DataFrame with metric names
        metric_names = df.columns.tolist()
        results_df = pd.DataFrame({
            "metric": metric_names,
            "mean": mean_metrics,
            "std": std_metrics
        })

        # Save to CSV
        results_df.to_csv(output_file, index=False)  

    # --- make boxplots ---
    # Example: fix one metric
    for chosen_metric in ["ECCE", "ECE", "MCE", "Brier", "NLL", "LCE", "MLCE"]:
        #chosen_metric = "ECE"  # replace with the metric you want to plot

        plt.figure(figsize=(8, 6))
        data = []
        labels = []
        for method, metrics_dict in collected_results.items():
            if chosen_metric in metrics_dict:
                data.append(metrics_dict[chosen_metric])
                labels.append(method)

        plt.boxplot(data, labels=labels, patch_artist=True)
        plt.ylabel(chosen_metric)
        plt.title(f"Comparison of {chosen_metric} across methods")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        os.makedirs(kwargs.save_path_boxplots, exist_ok=True)
        plot_file = os.path.join(kwargs.save_path_boxplots, f"{kwargs.data}_boxplot_{chosen_metric}.png")
        plt.savefig(plot_file, bbox_inches="tight")
        plt.close()
        
    


    