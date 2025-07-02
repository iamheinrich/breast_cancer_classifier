#!/bin/bash

NUM_PROCESSES=10
DEVICE_TYPE='gpu'
NUM_EPOCHS=10
HEATMAP_BATCH_SIZE=100
GPU_INDEX=0
NUM_EXAM_SAMPLES=10

EXAMS_PATH="../../Data/sample_data_omi_db/IMAGES"
DATA_PATH="../../Data/sample_data_omi_db/DATA"

PATCH_MODEL_PATH='models/sample_patch_model.p'
IMAGE_MODEL_PATH='models/sample_image_model.p'
IMAGEHEATMAPS_MODEL_PATH='models/sample_imageheatmaps_model.p'

SAMPLE_INPUT_PATH='omi_db_data'
SAMPLE_OUTPUT_PATH='omi_db_output'
export PYTHONPATH=$(pwd):$PYTHONPATH

echo 'Stage 0: Generate exam list'
python3 src/data_loading/optimam_preprocessor.py \
    --exams-path $EXAMS_PATH \
    --data-path $DATA_PATH \
    --sample-input-path $SAMPLE_INPUT_PATH \
    --num-exam-samples $NUM_EXAM_SAMPLES \
#    --annotations-path $ANNOTATIONS_PATH \

echo 'Stage 1: Crop Mammograms'
python3 src/cropping/crop_mammogram.py \
    --input-data-folder ${SAMPLE_INPUT_PATH}/images \
    --output-data-folder ${SAMPLE_OUTPUT_PATH}/cropped_images \
    --exam-list-path ${SAMPLE_INPUT_PATH}/exam_list_before_cropping.pkl \
    --cropped-exam-list-path ${SAMPLE_OUTPUT_PATH}/cropped_images/cropped_exam_list.pkl  \
    --num-processes $NUM_PROCESSES

echo 'Stage 2: Extract Centers'
python3 src/optimal_centers/get_optimal_centers.py \
    --cropped-exam-list-path ${SAMPLE_OUTPUT_PATH}/cropped_images/cropped_exam_list.pkl \
    --data-prefix ${SAMPLE_OUTPUT_PATH}/cropped_images \
    --output-exam-list-path ${SAMPLE_OUTPUT_PATH}/data.pkl \
    --num-processes $NUM_PROCESSES

echo 'Stage 3: Generate Heatmaps'
python3 src/heatmaps/run_producer.py \
    --model-path $PATCH_MODEL_PATH \
    --data-path ${SAMPLE_OUTPUT_PATH}/data.pkl \
    --image-path ${SAMPLE_OUTPUT_PATH}/cropped_images \
    --batch-size $HEATMAP_BATCH_SIZE \
    --output-heatmap-path ${SAMPLE_OUTPUT_PATH}/heatmaps \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_INDEX

echo 'Stage 4: Run Classifier (Image+Heatmaps)'
python3 src/modeling/run_model.py \
    --model-path $IMAGEHEATMAPS_MODEL_PATH \
    --data-path ${SAMPLE_OUTPUT_PATH}/data.pkl \
    --image-path ${SAMPLE_OUTPUT_PATH}/cropped_images \
    --output-path ${SAMPLE_OUTPUT_PATH}/imageheatmaps_predictions.csv \
    --use-heatmaps \
    --heatmaps-path ${SAMPLE_OUTPUT_PATH}/heatmaps \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_INDEX

#echo 'Stage 5: Evaluate Performance'
#python3 src/evaluation/calculate_auc.py \
#    --prediction-path ${SAMPLE_OUTPUT_PATH}/imageheatmaps_predictions.csv \
#    --label-path ${SAMPLE_INPUT_PATH}/exam_labels.pkl \
#    --performance-measures-path ${SAMPLE_OUTPUT_PATH}/performance_measures.csv
