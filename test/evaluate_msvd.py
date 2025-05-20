import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm
from openai import OpenAI
client = OpenAI(api_key="OPENAI_API_KEY")

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--ground_truth", type=str, default="../utils/msvd_test.jsonl")
    parser.add_argument("--pred_path", type=str, default="../exp_log")
    parser.add_argument("--method", type=str, default="qwen2", choices=["qwen", "qwen-uts", "qwen2", "qwen2-uts", "llava", "llava-uts"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")
    parser.add_argument("--generate_annotation", action="store_true", help="Generate annotations")
    parser.add_argument("--combine", action="store_true", help="Combine annotations")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate annotations")
    parser.add_argument("--merge", action="store_true", help="Merge annotations")
    parser.add_argument("--merge_list", type=str, default=None, help="List to merge")
    args = parser.parse_args()
    return args


def annotate(prediction_set, caption_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    for file in tqdm(caption_files):
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']
        try:
            # Compute the correctness score
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": 
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                    }
                ],
                temperature=0,  # Disable randomness by setting temperature to 0
                top_p=1,  # Use the full probability distribution, which is standard behavior
                n=1  # Generate a single completion
            )
            # Convert response to a Python dictionary.
            response_message = completion.choices[0].message.content
            print(response_message)
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")


def generate_annotation(args):
    # Load the prediction file
    pred_file_name = f"{args.method}_{args.start}_{args.end}.json"
    pred_file_path = os.path.join(args.pred_path, pred_file_name)
    with open(pred_file_path, "r") as file:
        file = open(pred_file_path, "r")
        pred_contents = json.load(file)

    # Load ground truth file
    ground_truth = {}
    with open(args.ground_truth, "r") as file:
        lines = file.readlines()
    for id, line in enumerate(lines):
        data = json.loads(line.strip())
        ground_truth[str(id)] = data

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for id, pred in pred_contents.items():
        assert id in ground_truth, f"ID {id} not found in ground truth"
        question = ground_truth[id]['question']
        answer = ground_truth[id]['answer']
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    output_dir = os.path.join(args.pred_path, f"{args.method}_annotation")
    os.makedirs(output_dir, exist_ok=True)
    caption_files = [f'{key}.json' for key in prediction_set.keys()]
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")


def combine_annotations(args):
    # Combine all the processed files into one
    combined_contents = {}
    output_dir = os.path.join(args.pred_path, f"{args.method}_annotation")

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                file_name = file_name.split(".")[0]
                combined_contents[file_name] = content

    # Write combined content to a json file
    json_path = os.path.join(args.pred_path, f"{args.method}_combined_annotations.json")
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file, indent=4)
    print("All evaluation completed!")


def evaluate_annotations(args):
    # Load the combined annotations
    json_path = os.path.join(args.pred_path, f"{args.method}_combined_annotations.json")
    with open(json_path, "r") as json_file:
        combined_contents = json.load(json_file)

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
        
    for key, result in combined_contents.items():
        # Computing score
        count += 1
        score_match = result[0]['score']
        score = int(score_match)
        score_sum += score

        try:
            # Computing accuracy
            pred = result[0]['pred']
        except:
            pred = result[0]['predicted']
        
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1
        else:
            raise Exception("Invalid prediction")
    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)


def merge_annotations(args):
    idx_list = args.merge_list.split(",")
    total_json = {}
    for idx in range(len(idx_list) - 1):
        pred_file_name = f"{args.method}_{idx_list[idx]}_{idx_list[idx+1]}.json"
        pred_file_path = os.path.join(args.pred_path, pred_file_name)
        with open(pred_file_path, "r") as json_file:
            total_json.update(json.load(json_file))

    json_path = f"{args.method}_0_0.json"
    json_path = os.path.join(args.pred_path, json_path)
    with open(json_path, "w") as json_file:
        json.dump(total_json, json_file, indent=4)
    print("Merge completed!")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    if args.merge:
        merge_annotations(args)

    if args.generate_annotation:
        generate_annotation(args)
    
    if args.combine:
        combine_annotations(args)
    
    if args.evaluate:
        evaluate_annotations(args)


if __name__ == "__main__":
    main()