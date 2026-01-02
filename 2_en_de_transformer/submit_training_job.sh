#!/bin/bash

#PBS -N transformer_TLA_training
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:mem=32G:ncpus=8:ngpus=1

# Training configuration - edit these as needed
EPOCHS=20
BATCH_SIZE=32
MODEL_NAME="transformer_TLA"

# Data paths (on cluster)
DATA_BASE="/mnt/lustre/helios-home/morovkat/hiero-transformer"
TRAIN_SRC="${DATA_BASE}/training_data/source_egy2tnt_cleaned.txt"
TRAIN_TGT="${DATA_BASE}/training_data/target_egy2tnt_cleaned.txt"
VAL_SRC="${DATA_BASE}/test_and_validation_data/validation_source_egy2tnt_cleaned.txt"
VAL_TGT="${DATA_BASE}/test_and_validation_data/validation_target_egy2tnt_cleaned.txt"
TEST_SRC="${DATA_BASE}/test_and_validation_data/test_source_egy2tnt_cleaned.txt"
TEST_TGT="${DATA_BASE}/test_and_validation_data/test_target_egy2tnt_cleaned.txt"

# Model settings (relative paths, will be resolved after cd)
MODEL_SETTINGS="model_settings_model7.json"
COMPILE_SETTINGS="model_compile_settings.json"

cd $PBS_O_WORKDIR

module load cuda/12.4.1

source /mnt/lustre/helios-home/morovkat/miniconda3/etc/profile.d/conda.sh
conda activate hiero-transformer

cd /mnt/lustre/helios-home/morovkat/hiero-transformer/2_en_de_transformer

# Build command with arguments
CMD="python s2s_transformer_pipeline.py \
    --model-settings ${MODEL_SETTINGS} \
    --compile-settings ${COMPILE_SETTINGS} \
    --train-src ${TRAIN_SRC} \
    --train-tgt ${TRAIN_TGT} \
    --val-src ${VAL_SRC} \
    --val-tgt ${VAL_TGT} \
    --test-src ${TEST_SRC} \
    --test-tgt ${TEST_TGT} \
    --model-name ${MODEL_NAME} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --new-class-dict"

echo "=========================================="
echo "Training Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Model Settings: ${MODEL_SETTINGS}"
echo ""
echo "Data Files:"
echo "  Train Source: ${TRAIN_SRC}"
echo "  Train Target: ${TRAIN_TGT}"
echo "  Val Source: ${VAL_SRC}"
echo "  Val Target: ${VAL_TGT}"
echo "  Test Source: ${TEST_SRC}"
echo "  Test Target: ${TEST_TGT}"
echo "=========================================="
echo ""
echo "Running: $CMD"
echo ""

$CMD

