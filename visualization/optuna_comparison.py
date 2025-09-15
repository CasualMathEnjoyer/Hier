# Replace with your study names
study_names = {
    "TPE": "bakalarka_TPE_short_3",
    "GP": "bakalarka_GP_short_3",
    "Random": "bakalarka_Random_short_3",
    "Grid": "bakalarka_Grid_short_3"
}

storage = "sqlite:////home/katka/PycharmProjects/Hier/2_en_de_transformer/optuna_study.db"

"""
Comparison of multiple Optuna studies (different optimizers) for maximization (accuracy).
Plots:
1. Convergence curves (max so far)
2. Distribution of accuracies
3. Best accuracy per optimizer (bar chart)
4. Accuracy per epoch with best points highlighted
"""

import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

studies = {name: optuna.load_study(study_name=s, storage=storage)
           for name, s in study_names.items()}

# Collect results
results = {}
for name, study in studies.items():
    values = [t.value for t in study.trials if t.value is not None]
    results[name] = values

# Convert to DataFrame for convenience
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))

# 1. Convergence curves (max so far)
plt.figure(figsize=(10, 6))
for name, values in results.items():
    best_so_far = []
    current_best = float("-inf")
    for v in values:
        if v > current_best:
            current_best = v
        best_so_far.append(current_best)
    plt.plot(best_so_far, label=name)

plt.title("Convergence Curves (Maximization)")
plt.xlabel("Trial")
plt.ylabel("Best Accuracy So Far")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Distribution of accuracies
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Distribution of Accuracy Values")
plt.xlabel("Optimizer")
plt.ylabel("Accuracy")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

def print_best_accuracy_latex_with_params(studies):
    """
    Print a LaTeX table showing, for each optimizer:
    - Best accuracy
    - Epoch it was first achieved
    - Hyperparameters of that trial

    Parameters:
        studies (dict): {optimizer_name: optuna.Study}
    """
    table_lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\begin{tabular}{l c c l}",
        "\\hline",
        "Optimizer & Best Accuracy & Epoch & Hyperparameters \\\\",
        "\\hline"
    ]

    for name, study in studies.items():
        # Only include completed trials
        completed_trials = [t for t in study.trials if t.value is not None]
        # Find best trial (max value)
        best_trial = max(completed_trials, key=lambda t: t.value)
        best_acc = best_trial.value
        epoch = best_trial.number + 1  # trials are 0-indexed
        # Convert hyperparameters dict to a string
        params_str = ", ".join(f"{k}={v}" for k, v in best_trial.params.items())
        table_lines.append(f"{name} & {best_acc:.4f} & {epoch} & {params_str} \\\\")

    table_lines.append("\\hline")
    table_lines.append("\\end{tabular}")
    table_lines.append("\\caption{Best accuracy per optimizer, epoch first achieved, and corresponding hyperparameters}")
    table_lines.append("\\end{table}")

    latex_table = "\n".join(table_lines)
    print(latex_table)

# Usage
print_best_accuracy_latex_with_params(studies)


# 4. Accuracy per epoch with best point highlighted
plt.figure(figsize=(10, 6))

for name, study in studies.items():
    values = [t.value for t in study.trials if t.value is not None]
    epochs = list(range(1, len(values) + 1))

    plt.plot(epochs, values, marker="o", linestyle="-", label=name, alpha=0.7)

    # Highlight the best (maximum) point
    best_idx = max(range(len(values)), key=lambda i: values[i])
    plt.scatter(
        epochs[best_idx], values[best_idx],
        color="red", s=120, edgecolors="black", zorder=5, marker="*"
    )

plt.title("Optimizer Accuracy per Epoch")
plt.xlabel("Epoch (Trial)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print best results
print("Best accuracy per optimizer:")
for name, best in best_results.items():
    print(f"{name}: {best:.4f}")
