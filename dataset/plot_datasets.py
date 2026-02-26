import os
import torch
import matplotlib.pyplot as plt
from load_dataset import SeisBenchPipelineWrapper

configs_to_test = [

    {"dataset_name": "STEAD", "split": "train", "model_type": "eqtransformer", "max_distance": 300, "transformation_shape": "triangle", "transformation_sigma": 20},
    {"dataset_name": "STEAD", "split": "train", "model_type": "phasenet", "max_distance":100, "transformation_shape": "gaussian", "transformation_sigma": 10},
    {"dataset_name": "STEAD", "split": "train", "model_type": "unet", "max_distance": None, "transformation_shape": "gaussian", "transformation_sigma": 10},
    {"dataset_name": "INSTANCE", "split": "train", "model_type": "eqtransformer", "max_distance": 300, "transformation_shape": "triangle", "transformation_sigma": 20},
    {"dataset_name": "INSTANCE", "split": "train", "model_type": "phasenet", "max_distance": 100, "transformation_shape": "gaussian", "transformation_sigma": 10},
    {"dataset_name": "INSTANCE", "split": "train", "model_type": "unet", "max_distance": None, "transformation_shape": "gaussian", "transformation_sigma": 10},
    {"dataset_name": "STEAD", "split": "test", "model_type": "eqtransformer", "max_distance": 300, "transformation_shape": "triangle", "transformation_sigma": 20},
    {"dataset_name": "STEAD", "split": "test", "model_type": "phasenet", "max_distance":100, "transformation_shape": "gaussian", "transformation_sigma": 10},
    {"dataset_name": "STEAD", "split": "test", "model_type": "unet", "max_distance": None, "transformation_shape": "gaussian", "transformation_sigma": 10},
    {"dataset_name": "STEAD", "split": "dev", "model_type": "eqtransformer", "max_distance": 300, "transformation_shape": "triangle", "transformation_sigma": 20},
    {"dataset_name": "VCSEIS", "split": "train", "model_type": "eqtransformer", "max_distance": 300, "transformation_shape": "triangle", "transformation_sigma": 20},
    {"dataset_name": "GEOFON", "split": "train", "model_type": "eqtransformer", "max_distance": 300, "transformation_shape": "triangle", "transformation_sigma": 20},
    {"dataset_name": "TXED", "split": "train", "model_type": "eqtransformer", "max_distance": 300, "transformation_shape": "triangle", "transformation_sigma": 20},
]

def plot_batch(batch, config, save_path):
    print(f"\n--- Plotting Example for {config['model_type']} ({config['dataset_name']}) ---")
    
    # Check if we have y_det
    has_det = "y_det" in batch
    subplots_count = 4 if has_det else 3
    num_samples = min(9, batch["X"].shape[0])
    
    fig, axs = plt.subplots(subplots_count, num_samples, figsize=(4 * num_samples, 2.5 * subplots_count), sharex='col', squeeze=False)
    
    for i in range(num_samples):
        # Extract the i-th sample
        X_sample = batch["X"][i].numpy()
        y_p_sample = batch["y_p"][i].numpy()
        y_s_sample = batch["y_s"][i].numpy()
        
        # Plot waveforms (3 components)
        for c, comp in enumerate("ZNE"):
            if c < len(X_sample):
                axs[0, i].plot(X_sample[c], label=f"Comp {comp}")
        if i == 0:
            axs[0, i].set_ylabel("Amplitude")
            axs[0, i].legend(loc="upper right")
            
        axs[0, i].set_title(f"Sample {i+1}")
                
        # Plot P-wave probability
        axs[1, i].plot(y_p_sample[0], color="red", label="P-wave target")
        if i == 0:
            axs[1, i].set_ylabel("Probability")
            axs[1, i].legend(loc="upper right")
        
        # Plot S-wave probability
        axs[2, i].plot(y_s_sample[0], color="blue", label="S-wave target")
        if i == 0:
            axs[2, i].set_ylabel("Probability")
            axs[2, i].legend(loc="upper right")
        
        # Plot Detection probability
        if has_det:
            y_det_sample = batch["y_det"][i].numpy()
            axs[3, i].plot(y_det_sample[0], color="green", label="Detection target")
            if i == 0:
                axs[3, i].set_ylabel("Probability")
                axs[3, i].legend(loc="upper right")

    # Set x-labels on the last row
    for i in range(num_samples):
        if not has_det:
            axs[2, i].set_xlabel("Time samples")
        else:
            axs[3, i].set_xlabel("Time samples")

    fig.suptitle(f"{config['model_type'].upper()}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path)
    print(f"Saved plot to '{save_path}'")
    plt.close(fig)

if __name__ == "__main__":
    if not os.path.exists("test_outputs"):
        os.makedirs("test_outputs")

    for i, config in enumerate(configs_to_test):
        print(f"\n==========================================")
        print(f"Testing Config {i+1}/{len(configs_to_test)}: {config}")
        
        try:
            pipeline = SeisBenchPipelineWrapper(**config)
            loader = pipeline.get_dataloader(batch_size=32, num_workers=0, shuffle=True)
            
            final_batch = {"X": [], "y_p": [], "y_s": []}
            has_det = "eqtransformer" in config['model_type']
            if has_det:
                final_batch["y_det"] = []
                
            # Iterate through batches to find exactly 9 samples containing events
            for b_idx, batch in enumerate(loader):
                for i in range(batch["X"].shape[0]):
                    # If target probability exceeds 0.1, we assume it's an event trace
                    if batch["y_p"][i].max() > 0.1 or batch["y_s"][i].max() > 0.1:
                        final_batch["X"].append(batch["X"][i])
                        final_batch["y_p"].append(batch["y_p"][i])
                        final_batch["y_s"].append(batch["y_s"][i])
                        if has_det:
                            final_batch["y_det"].append(batch["y_det"][i])
                            
                    if len(final_batch["X"]) == 9:
                        break
                if len(final_batch["X"]) == 9:
                    break
            
            if len(final_batch["X"]) < 9:
                print(f"Warning: Only found {len(final_batch['X'])} events.")
                
            if len(final_batch["X"]) > 0:
                stacked_batch = {k: torch.stack(v) for k, v in final_batch.items()}
                
                print(f"Batch keys available: {list(stacked_batch.keys())}")
                print(f"Waveform shape (X): {stacked_batch['X'].shape}")
                print(f"P-wave shape (y_p): {stacked_batch['y_p'].shape}")
                print(f"S-wave shape (y_s): {stacked_batch['y_s'].shape}")
                if "y_det" in stacked_batch:
                    print(f"Det shape (y_det): {stacked_batch['y_det'].shape}")
                    
                save_name = f"test_outputs/plot_{config['dataset_name']}_{config['model_type']}_{config['split']}.png"
                plot_batch(stacked_batch, config, save_name)
            else:
                print("No events found in the inspected batches.")
            
        except Exception as e:
            print(f"Error testing configuration {config}: {e}")
            import traceback
            traceback.print_exc()
