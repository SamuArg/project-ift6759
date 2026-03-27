import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Input sizes
input_sizes = [25, 50, 100, 200]

# Model results
stead_model = {
    "MAE": [0.4507, 0.3943, 0.4555, 0.3224],
    "R2": [0.5991, 0.6851, 0.6114, 0.7883],
}

instance_model = {
    "MAE": [0.4374, 0.4064, 0.3825, 0.3578],
    "R2": [0.3061, 0.3976, 0.4710, 0.5245],
}

stead_model_coords = {
    "MAE": [0.3647, 0.3377, 0.3106, 0.2833],
    "R2": [0.7381, 0.7768, 0.8114, 0.8442],
}

instance_model_coords = {
    "MAE": [0.4166, 0.3872, 0.3614, 0.3388],
    "R2": [0.3685, 0.4501, 0.5207, 0.5778],
}

# Baselines
stead_baseline = {"MAE":0.7658, "R2":0}
instance_baseline = {"MAE":0.5115, "R2":0}

# Build dataframe
rows = []
for i, size in enumerate(input_sizes):
    rows.append({"dataset":"STEAD","type":"Model","size":size,"MAE":stead_model["MAE"][i],"R2":stead_model["R2"][i]})
    rows.append({"dataset":"STEAD","type":"Model+Coords","size":size,"MAE":stead_model_coords["MAE"][i],"R2":stead_model_coords["R2"][i]})
    rows.append({"dataset":"INSTANCE","type":"Model","size":size,"MAE":instance_model["MAE"][i],"R2":instance_model["R2"][i]})
    rows.append({"dataset":"INSTANCE","type":"Model+Coords","size":size,"MAE":instance_model_coords["MAE"][i],"R2":instance_model_coords["R2"][i]})

df = pd.DataFrame(rows)

sns.set_theme(style="whitegrid", context="paper")
palette = sns.color_palette("colorblind")

# Create 2x2 figure
fig, axes = plt.subplots(2,2, figsize=(10,8))

# Map axes manually to have INSTANCE MAE top-right
ax_map = {
    ("STEAD","MAE"): axes[0,0],
    ("INSTANCE","MAE"): axes[0,1],
    ("STEAD","R2"): axes[1,0],
    ("INSTANCE","R2"): axes[1,1],
}

datasets = ["STEAD", "INSTANCE"]
metrics = ["MAE", "R2"]

for dataset in datasets:
    for metric in metrics:
        ax = ax_map[(dataset, metric)]
        data = df[df.dataset == dataset]
        model = data[data.type=="Model"]
        coords = data[data.type=="Model+Coords"]

        # Scatter points
        ax.scatter(model["size"], model[metric], marker="o",
                   color=palette[0] if dataset=="STEAD" else palette[1], label="Model")
        ax.scatter(coords["size"], coords[metric], marker="o", facecolors="white",
                   edgecolors=palette[0] if dataset=="STEAD" else palette[1], label="Model + Coords")

        # Vertical lines between Model and Model+Coords of same size
        for size in input_sizes:
            base = model[model["size"]==size]
            coord = coords[coords["size"]==size]
            if not base.empty and not coord.empty:
                ax.plot([size, size], [base[metric].values[0], coord[metric].values[0]],
                        linestyle="--", color=palette[0] if dataset=="STEAD" else palette[1], alpha=0.7)

        # Add baseline as horizontal line
        baseline = stead_baseline[metric] if dataset=="STEAD" else instance_baseline[metric]
        ax.hlines(y=baseline, xmin=min(input_sizes)-5, xmax=max(input_sizes)+5,
                  colors='gray', linestyles=':', label="Baseline")

        ax.set_title(f"{dataset} — {metric}")
        ax.set_xlabel("Input Size")
        ax.set_xlim(min(input_sizes)-5, max(input_sizes)+5)
        ax.set_ylim(0,1)
        if metric=="MAE":
            ax.set_ylabel("MAE")
        else:
            ax.set_ylabel("R2")
        ax.grid(True, alpha=0.3)

# Add legend to top-left plot (STEAD MAE)
axes[0,0].legend()

plt.tight_layout()
plt.savefig("magnitude_results_4plots_baseline.png", dpi=300)
plt.show()