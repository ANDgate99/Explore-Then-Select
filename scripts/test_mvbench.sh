USE_CUDA=$1 # 0,1
METHOD=$2 # [llava, llava-uts, qwen2, qwen2-uts]
START=$3 # 0
END=$4 # 0
FRAMES=$5 # 256
BUDGET_FRAMES=$6 # 64
OUTPUT=$7 # exp_log
DATASET=( \
         'action_sequence' \
         'moving_count' \
         'action_prediction' \
         'action_count' \
         'scene_transition' \
         'object_shuffle' \
         'object_existence'
        #  'fine_grained_pose' \
         'unexpected_action' \ 
         'moving_direction' \
         'state_change' \
         'object_interaction' \
         'character_order' \
         'action_localization' \
         'counterfactual_inference' \
         'fine_grained_action' \
         'moving_attribute' \
         'egocentric_navigation' \
         'action_antonym' \
         'episodic_reasoning' \
        )

echo "========================="
echo "Use cuda:" $USE_CUDA
echo "Use model:" $METHOD
echo "Use dataset:"
for dataset in ${DATASET[@]}; do
    echo $dataset
done
echo "Test start:" $START
echo "Test end:" $END
echo "Test frames:" $FRAMES
echo "Test budget frames:" $BUDGET_FRAMES
echo "Output:" $OUTPUT
echo "========================="

cd ..

if [[ $METHOD == llava* ]]; then
    echo "Using llava"
    for dataset in ${DATASET[@]}; do
        echo $dataset
        CUDA_VISIBLE_DEVICES=$USE_CUDA python -m test.test_mvbench_llava \
            --frames $FRAMES \
            --budget_frames $BUDGET_FRAMES \
            --type $dataset \
            --method $METHOD \
            --start $START \
            --end $END \
            --output $OUTPUT
            # --model /path/to/model \
            # --dataset /path/to/dataset \
            # --video_path /path/to/video_folder \
    done
else
    echo "Using qwen2 vl"
    for dataset in ${DATASET[@]}; do
        echo $dataset
        CUDA_VISIBLE_DEVICES=$USE_CUDA python -m test.test_mvbench_qwen \
            --frames $FRAMES \
            --budget_frames $BUDGET_FRAMES \
            --type $dataset \
            --method $METHOD \
            --start $START \
            --end $END \
            --output $OUTPUT
            # --model /path/to/model \
            # --dataset /path/to/dataset \
            # --video_path /path/to/video_folder \
    done
fi