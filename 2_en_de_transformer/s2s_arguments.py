import json
import argparse
from s2s_transformer_pipeline import run_model_pipeline

def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load JSON settings from specified files.")
    parser.add_argument("--model_settings", type=str, required=True, help="Path to model settings JSON file.")
    parser.add_argument("--model_compile_settings", type=str, required=True, help="Path to model compile settings JSON file.")
    parser.add_argument("--run_settings", type=str, required=True, help="Path to run settings JSON file.")

    args = parser.parse_args()

    model_settings = load_json(args.model_settings)
    model_compile_settings = load_json(args.model_compile_settings)
    run_settings = load_json(args.run_settings)

    print("\nSettings Overview:\n")
    print(f"{'Category':<25}{'Details'}")
    print("-" * 50)
    print(f"{'Model Settings':<25}{model_settings}")
    print(f"{'Model Compile Settings':<25}{model_compile_settings}")
    print(f"{'Run Settings':<25}{run_settings}")

    run_model_pipeline(model_settings, model_compile_settings, run_settings)

    # avg_RLEV_per_valid_char | 0.339504
    # avg_RLEV_per_pred_char | 0.3592018

    # avg_RLEV_per_valid_char        | 0.3409012
    # avg_RLEV_per_pred_char         | 0.3601476

    # avg_RLEV_per_valid_char | 0.3409012
    # avg_RLEV_per_pred_char | 0.3601476


