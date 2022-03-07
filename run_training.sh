#!/bin/bash
MIN_PRODUCTS=0
RAW_OUTPUT=/workspace/datasets/categories/output
INPUT_DIR=./data/pruned_products/
SHUFFLED_OUTPUT=/workspace/datasets/categories/output.shuffled
TRAINING_DATA=/workspace/datasets/categories/training_data
TESTING_DATA=/workspace/datasets/categories/testing_data
FASTTEXT=~/fastText-0.9.2/fasttext
MODEL=/workspace/datasets/categories/model
MODEL_BIN=$MODEL.bin
CREATE_DATA_ARGS="--stem"
TRAIN_MODEL_ARGS=

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--min-products)
            MIN_PRODUCTS=$2
            shift
            shift
            ;;
        -cd|--create-data-args)
            CREATE_DATA_ARGS=$2
            shift
            shift
            ;;
        -tm|--train-model-args)
            TRAIN_MODEL_ARGS=$2
            shift
            shift
            ;;
    esac
done

echo "Creating training data..."
python week3/createContentTrainingData.py --input $INPUT_DIR --output $RAW_OUTPUT --min_products $MIN_PRODUCTS $CREATE_DATA_ARGS

echo "Shuffling..."
shuf $RAW_OUTPUT -o $SHUFFLED_OUTPUT

echo "Grabbing training set..."
head -n 10000 $SHUFFLED_OUTPUT > $TRAINING_DATA

echo "Grabbing testing set..."
tail -n 10000 $SHUFFLED_OUTPUT > $TESTING_DATA

echo ""
echo "Default Training"
echo "-------------"
$FASTTEXT supervised -input $TRAINING_DATA -output $MODEL $TRAIN_MODEL_ARGS
echo "Top 1"
$FASTTEXT test $MODEL_BIN $TESTING_DATA
echo "Top 5"
$FASTTEXT test $MODEL_BIN $TESTING_DATA 5

echo ""
echo "Finer Training"
echo "-------------"
$FASTTEXT supervised -input $TRAINING_DATA -output $MODEL $TRAIN_MODEL_ARGS -epoch 25 -wordNgrams 2 -lr 1
echo "Top 1"
$FASTTEXT test $MODEL_BIN $TESTING_DATA
echo "Top 5"
$FASTTEXT test $MODEL_BIN $TESTING_DATA 5