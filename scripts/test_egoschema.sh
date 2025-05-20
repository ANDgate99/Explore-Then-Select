USE_CUDA=$1 # 0,1
METHOD=$2 # [llava, llava-uts, qwen2, qwen2-uts]
DATASET=$3 # [Subset, MC] 
START=$4 # 0
END=$5 # 0
FRAMES=$6 # 256
BUDGET_FRAMES=$7 # 64
OUTPUT=$8 # exp_log

echo "========================="
echo "Use cuda:" $USE_CUDA
echo "Use model:" $METHOD
echo "Use dataset:" $DATASET
echo "Test start:" $START
echo "Test end:" $END
echo "Test frames:" $FRAMES
echo "Test budget frames:" $BUDGET_FRAMES
echo "Output:" $OUTPUT
echo "========================="

cd ..

if [[ $METHOD == llava* ]]; then
    echo "Using llava"
    CUDA_VISIBLE_DEVICES=$USE_CUDA python -m test.test_egoschema_llava \
        --frames $FRAMES \
        --budget_frames $BUDGET_FRAMES \
        --type $DATASET \
        --method $METHOD \
        --start $START \
        --end $END \
        --output $OUTPUT
        # --model /path/to/model \
        # --dataset /path/to/dataset \
        # --video_path /path/to/video_folder \
elif [[ $METHOD == qwen2* ]]; then
    echo "Using qwen2 vl"
    CUDA_VISIBLE_DEVICES=$USE_CUDA python -m test.test_egoschema_qwen \
        --frames $FRAMES \
        --budget_frames $BUDGET_FRAMES \
        --type $DATASET \
        --method $METHOD \
        --start $START \
        --end $END \
        --output $OUTPUT
        # --model /path/to/model \
        # --dataset /path/to/dataset \
        # --video_path /path/to/video_folder \
elif [[ $METHOD == qwen* ]]; then
    echo "Using qwen2.5 vl"
    CUDA_VISIBLE_DEVICES=$USE_CUDA python -m test.test_egoschema_qwen_2_5 \
        --frames $FRAMES \
        --budget_frames $BUDGET_FRAMES \
        --type $DATASET \
        --method $METHOD \
        --start $START \
        --end $END \
        --output $OUTPUT
        # --model /path/to/model \
        # --dataset /path/to/dataset \
        # --video_path /path/to/video_folder \
fi