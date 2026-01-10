#!/bin/bash

# Script to submit training jobs for the three best models from free search optimization
# Creates modified run_settings files for each model and submits PBS jobs

cd /home/katka/PycharmProjects/Hier/2_en_de_transformer

# Base run settings file
BASE_RUN_SETTINGS="run_settings_trainFreeSearchModel.json"

# Model configurations
declare -A MODELS=(
    ["tpe"]="model_settings_tpe_acc0.979482_trial41.json"
    ["gp"]="model_settings_gp_acc0.980199_trial38.json"
    ["random"]="model_settings_random_acc0.979225_trial33.json"
)

# Model short names (used in run_settings)
declare -A MODEL_NAMES=(
    ["tpe"]="model_free_search_tpe"
    ["gp"]="model_free_search_gp"
    ["random"]="model_free_search_random"
)

# Check if base run settings exists
if [ ! -f "$BASE_RUN_SETTINGS" ]; then
    echo "Error: Base run settings file not found: $BASE_RUN_SETTINGS"
    exit 1
fi

# Function to create modified run_settings file
create_run_settings() {
    local optimizer=$1
    local model_name=$2
    local output_file="run_settings_${optimizer}_free_search.json"
    
    # Use Python to modify the JSON file
    python3 << EOF
import json

with open("$BASE_RUN_SETTINGS", 'r') as f:
    settings = json.load(f)

settings["model_name_short"] = "$model_name"

with open("$output_file", 'w') as f:
    json.dump(settings, f, indent=2)

print(f"Created: $output_file with model_name_short: $model_name")
EOF
}

# Function to create and submit PBS job
submit_job() {
    local optimizer=$1
    local model_settings=$2
    local run_settings="run_settings_${optimizer}_free_search.json"
    local compile_settings="model_compile_settings.json"
    
    # Create PBS script content
    local pbs_script="submit_training_${optimizer}_free_search.pbs"
    
    cat > "$pbs_script" << EOF
#!/bin/bash

#PBS -N gpu_transformer_${optimizer}_free_search
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:mem=64G:ncpus=16:ngpus=1

cd \$PBS_O_WORKDIR

module load cuda/12.4.1

# Activate H_keras3 virtualenv
source /mnt/lustre/helios-home/morovkat/H_keras3/bin/activate

cd /mnt/lustre/helios-home/morovkat/H_optimize/2_en_de_transformer

# Model settings files
MODEL_SETTINGS="${model_settings}"
COMPILE_SETTINGS="${compile_settings}"
RUN_SETTINGS="${run_settings}"

# Build command with arguments
CMD="python s2s_transformer_pipeline.py \\
    --model-settings \${MODEL_SETTINGS} \\
    --compile-settings \${COMPILE_SETTINGS} \\
    --run-settings \${RUN_SETTINGS}"

echo "=========================================="
echo "Training Configuration:"
echo "  Optimizer: ${optimizer}"
echo "  Model Settings: \${MODEL_SETTINGS}"
echo "  Compile Settings: \${COMPILE_SETTINGS}"
echo "  Run Settings: \${RUN_SETTINGS}"
echo "=========================================="
echo ""
echo "Running: \$CMD"
echo ""

\$CMD
EOF

    chmod +x "$pbs_script"
    echo "Created PBS script: $pbs_script"
    
    # Submit the job
    echo "Submitting job for ${optimizer} optimizer..."
    qsub "$pbs_script"
    echo "Job submitted for ${optimizer}"
    echo ""
}

# Main execution
echo "=========================================="
echo "Preparing training jobs for free search models"
echo "=========================================="
echo ""

# Create run_settings files for each model
echo "Creating run_settings files..."
for optimizer in "${!MODELS[@]}"; do
    model_name="${MODEL_NAMES[$optimizer]}"
    create_run_settings "$optimizer" "$model_name"
done

echo ""
echo "=========================================="
echo "Submitting training jobs"
echo "=========================================="
echo ""

# Submit jobs for each model
for optimizer in "${!MODELS[@]}"; do
    model_settings="${MODELS[$optimizer]}"
    
    # Check if model settings file exists
    if [ ! -f "$model_settings" ]; then
        echo "Warning: Model settings file not found: $model_settings"
        continue
    fi
    
    submit_job "$optimizer" "$model_settings"
done

echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Submitted jobs:"
for optimizer in "${!MODELS[@]}"; do
    echo "  - ${optimizer}: gpu_transformer_${optimizer}_free_search"
done
echo ""
echo "Check job status with: qstat"

