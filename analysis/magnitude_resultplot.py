import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

input_sizes = [25, 50, 100, 200]

rows = []

# Old model results (no coords)
stead_model = {
    "MSE":[0.3811,0.2993,0.3694,0.2012],
    "MAE":[0.4507,0.3943,0.4555,0.3224],
    "RMSE":[0.6173,0.5471,0.6078,0.4485],
    "R2":[0.5991,0.6851,0.6114,0.7883]
}

instance_model = {
    "MSE":[0.3361,0.2917,0.2562,0.2303],
    "MAE":[0.4374,0.4064,0.3825,0.3578],
    "RMSE":[0.5797,0.5401,0.5062,0.4799],
    "R2":[0.3061,0.3976,0.4710,0.5245]
}

# New model results (with coords) — from magntiude_results.txt
stead_model_coords = {
    "MSE": [0.2489, 0.2121, 0.1793, 0.1481],
    "MAE": [0.3647, 0.3377, 0.3106, 0.2833],
    "RMSE":[0.4989, 0.4606, 0.4234, 0.3848],
    "R2":  [0.7381, 0.7768, 0.8114, 0.8442]
}

instance_model_coords = {
    "MSE": [0.3058, 0.2663, 0.2321, 0.2045],
    "MAE": [0.4166, 0.3872, 0.3614, 0.3388],
    "RMSE":[0.5530, 0.5160, 0.4818, 0.4522],
    "R2":  [0.3685, 0.4501, 0.5207, 0.5778]
}

stead_baseline = {"MSE":0.9505,"MAE":0.7658,"RMSE":0.9749,"R2":0}
instance_baseline = {"MSE":0.4843,"MAE":0.5115,"RMSE":0.6959,"R2":0}

# Build dataframe
for i, size in enumerate(input_sizes):
    for metric in stead_model:
        rows.append({"input_size": size, "dataset": "STEAD",    "type": "Model",        "metric": metric, "value": stead_model[metric][i]})
        rows.append({"input_size": size, "dataset": "INSTANCE", "type": "Model",        "metric": metric, "value": instance_model[metric][i]})
        rows.append({"input_size": size, "dataset": "STEAD",    "type": "Model+Coords", "metric": metric, "value": stead_model_coords[metric][i]})
        rows.append({"input_size": size, "dataset": "INSTANCE", "type": "Model+Coords", "metric": metric, "value": instance_model_coords[metric][i]})
        rows.append({"input_size": size, "dataset": "STEAD",    "type": "Baseline",     "metric": metric, "value": stead_baseline[metric]})
        rows.append({"input_size": size, "dataset": "INSTANCE", "type": "Baseline",     "metric": metric, "value": instance_baseline[metric]})

df = pd.DataFrame(rows)

sns.set_theme(style="whitegrid", context="paper")
sns.set_palette("colorblind")

g = sns.relplot(
    data=df,
    x="input_size",
    y="value",
    hue="dataset",
    style="type",          # model vs model+coords vs baseline
    col="metric",
    kind="line",
    marker="o",
    col_wrap=2,
    height=4,
    facet_kws={"sharey": False}
)

g.set_axis_labels("Input Size (samples)", "Metric Value")
g.set_titles("{col_name}")
g.set(ylim=(-0.1, 1))

g.fig.suptitle("Model vs Model+Coords vs Baseline Performance", y=1.02)

g.savefig("magnitude_results.png", dpi=300, bbox_inches="tight")