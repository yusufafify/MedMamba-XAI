import nbformat

def update_nb():
    file_path = "/home/group2/youssef/MedMamba-XAI/notebooks/05_all_models_comparison.ipynb"
    with open(file_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Cell 10: Validation Curves (add DataFrame summary)
    if "display(styled_summary)" not in nb.cells[10].source:
        summary_code = """
from IPython.display import display, HTML

print("### Best Validation Performance Summary")
summary_pivot = best_val_df.pivot(index="dataset", columns="model_name", values=["best_val_accuracy", "best_val_f1_macro"])
styled_summary = summary_pivot.style.highlight_max(axis=1, color='lightgreen', props='font-weight:bold;')
display(styled_summary)
"""
        nb.cells[10].source += "\n" + summary_code

    # Cell 14: Confusion Matrix Display (add medmnist labels & xticks_rotation)
    if "import medmnist" not in nb.cells[14].source:
        cm_code = """model_names = list(MODEL_SPECS.keys())

fig, axes = plt.subplots(len(DATASETS), len(model_names), figsize=(14, 5 * len(DATASETS)), constrained_layout=True)
if len(DATASETS) == 1:
    axes = np.array([axes])

import medmnist

for i, dataset in enumerate(DATASETS):
    for j, model_name in enumerate(model_names):
        ax = axes[i, j]
        y_true, y_pred = prediction_cache[(model_name, dataset)]

        n_classes = DATASET_META[dataset]["num_classes"]
        labels = np.arange(n_classes)
        
        info = medmnist.INFO[dataset]
        label_dict = info["label"]
        target_names = [label_dict[str(k)] for k in range(n_classes)]
        
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False, xticks_rotation="vertical")

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        ax.set_title(f"{dataset} | {model_name}\\nAcc={acc:.4f}, F1={f1m:.4f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

plt.show()"""
        nb.cells[14].source = cm_code
        
    # Also add a summary table for test performance
    if "test_summary_pivot" not in nb.cells[14].source:
        test_summary_code = """
print("### Test Performance Summary")
test_summary_pivot = test_summary_df.pivot(index="dataset", columns="model_name", values=["test_accuracy", "test_f1_macro"])
styled_test_summary = test_summary_pivot.style.highlight_max(axis=1, color='lightgreen', props='font-weight:bold;')
display(styled_test_summary)
"""
        nb.cells[14].source += "\n" + test_summary_code

    # Cell 16: Ensure RECOMPUTE_EFFICIENCY = True
    nb.cells[16].source = nb.cells[16].source.replace("RECOMPUTE_EFFICIENCY = False", "RECOMPUTE_EFFICIENCY = True")

    # Cell 18: Add styles for explicitly highlighting efficiency comparison
    if "styled_efficiency" not in nb.cells[18].source:
        eff_code = """
print("### Efficiency Summary")
# Higher is worse for latency and params, so we highlight min values.
styled_efficiency = overall_comparison_df.style.highlight_min(subset=["params_M", "flops_G", "latency_mean_ms", "latency_p95_ms"], color='lightgreen', props='font-weight:bold;')
styled_efficiency = styled_efficiency.highlight_max(subset=["test_f1_macro"], color='lightblue', props='font-weight:bold;')
display(styled_efficiency)
"""
        nb.cells[18].source = nb.cells[18].source.replace("overall_comparison_df", "overall_comparison_df\n" + eff_code)

    with open(file_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    update_nb()
