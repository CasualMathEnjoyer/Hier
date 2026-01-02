#!/bin/bash

#PBS -N transformer_TLA_training
#PBS -l walltime=72:00:00
#PBS -q cpu_a
#PBS -j oe
#PBS -l select=1:mem=32G:ncpus=8:ngpus=1

cd $PBS_O_WORKDIR

module load cuda/12.4.1

source /mnt/lustre/helios-home/morovkat/miniconda3/etc/profile.d/conda.sh
conda activate hiero-transformer

cd /mnt/lustre/helios-home/morovkat/hiero-transformer/2_en_de_transformer

# Model settings files
MODEL_SETTINGS="model_settings_model7.json"
COMPILE_SETTINGS="model_compile_settings.json"
RUN_SETTINGS="run_settings_transformer_TLA.json"

# Build command with arguments
CMD="python s2s_transformer_pipeline.py \
    --model-settings ${MODEL_SETTINGS} \
    --compile-settings ${COMPILE_SETTINGS} \
    --run-settings ${RUN_SETTINGS}"

echo "=========================================="
echo "Training Configuration:"
echo "  Model Settings: ${MODEL_SETTINGS}"
echo "  Compile Settings: ${COMPILE_SETTINGS}"
echo "  Run Settings: ${RUN_SETTINGS}"
echo "=========================================="
echo ""
echo "Running: $CMD"
echo ""

$CMD

