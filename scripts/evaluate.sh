cd ../test

# python evaluate_msvd.py \
#     --ground_truth /path/to/data/file \
#     --pred_path /path/to/prediction/folder \
#     --method qwen2-uts \
#     --merge \
#     --merge_list 0,2000,...


python evaluate_msvd.py \
    --ground_truth /path/to/data/file \
    --pred_path /path/to/prediction/folder \
    --method qwen2-uts \
    --num_tasks 50 \
    --generate_annotation


python evaluate_msvd.py \
    --ground_truth /path/to/data/file \
    --pred_path /path/to/prediction/folder \
    --method qwen2-uts \
    --combine


python evaluate_msvd.py \
    --ground_truth /path/to/data/file \
    --pred_path /path/to/prediction/folder \
    --method qwen2-uts \
    --evaluate


