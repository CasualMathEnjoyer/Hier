#!/bin/bash

#PBS -N gpu_transformer_TLA_training
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:mem=32G:ncpus=8:ngpus=1

cd $PBS_O_WORKDIR

module load cuda/12.4.1

# H_keras3 is a directory, not a conda environment - add it to PYTHONPATH
export PYTHONPATH="/mnt/lustre/helios-home/morovkat/H_keras3:${PYTHONPATH}"

cd /mnt/lustre/helios-home/morovkat/H_optimize/2_en_de_transformer

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

