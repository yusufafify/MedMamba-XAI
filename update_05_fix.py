import nbformat

def update_nb_fix():
    file_path = "/home/group2/youssef/MedMamba-XAI/notebooks/05_all_models_comparison.ipynb"
    with open(file_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Re-write cell 18 correctly
    correct_cell_18 = """overall_comparison_df = test_summary_df.merge(
    efficiency_df[["dataset", "model_name", "params_M", "flops_G", "latency_mean_ms", "latency_p95_ms"]],
    on=["dataset", "model_name"],
    how="left",
)

overall_comparison_df = overall_comparison_df.sort_values(["dataset", "test_f1_macro"], ascending=[True, False])

from IPython.display import display, HTML
print("### Efficiency Summary")
# Higher is worse for latency and params, so we highlight min values.
styled_efficiency = overall_comparison_df.style.highlight_min(subset=["params_M", "flops_G", "latency_mean_ms", "latency_p95_ms"], color='lightgreen', props='font-weight:bold;')
styled_efficiency = styled_efficiency.highlight_max(subset=["test_f1_macro"], color='lightblue', props='font-weight:bold;')
display(styled_efficiency)
"""
    nb.cells[18].source = correct_cell_18

    with open(file_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    update_nb_fix()
