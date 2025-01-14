import json
import pdb
from tqdm import tqdm
import random
import re
import copy
import torch
import os
import sys
from typing import List
import fire

# For Qwen from Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM

###############################################################################
# Utilities
###############################################################################
def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_data(file_path):
    return json.load(open(file_path))

def read_data(file_path):
    raw_data = []
    with open(file_path, "r", encoding='utf-8') as fin:
        for line in fin:
            raw_data.append(json.loads(line))
    return raw_data

def extract_answers(data):
    answers = []
    for each in data:
        each_answer = int(each['answer'].split('\n#### ')[1].replace(",", ""))
        answers.append(each_answer)
    return answers

def extract_pred_answers(data, verbose=False):
    preds = []
    for each in data:
        each_pred = each.split('answer (arabic numerals) is ')[1].split('.\n')[0]
        each_pred = re.sub(r'[a-zA-Z%$=\-]', ' ', each_pred)  # remove alphabets and insert whitespace
        each_pred = each_pred.replace(",", "")  # remove commas
        each_pred = ' '.join(each_pred.split())  # remove multiple spaces
        each_pred = each_pred.split()[-1]
        try:
            each_pred = int(float(each_pred))
        except:
            each_pred = 0
        if verbose:
            print(each)
            print(each_pred)
            print("----")
        preds.append(each_pred)
    return preds

def extract_cot_with_example_pred_answers(preds):
    refined_preds = []
    all_preds = []
    for pred in preds:
        try:
            pred1 = pred.split("The answer is ")[9]
        except:
            pred1 = "234234"
        all_preds.append(pred1)
        pred2 = pred1.split("\n\nQuestion:")[0]
        pred3 = re.sub(r'[a-zA-Z%$=\-\.]', ' ', pred2)
        pred4 = pred3.replace(",", "")  # remove commas
        pred5 = ' '.join(pred4.split())  # remove multiple spaces
        try:
            pred6 = pred5.split()[-1]
            pred7 = int(float(pred6))
        except:
            pred7 = 234234
        refined_preds.append(pred7)
    return refined_preds, all_preds

###############################################################################
# Accuracy Computation
###############################################################################
def compute_accuracy(gold_data_path, pred_data_path, cot=False, save_path=None, verbose=True, sample=None):
    if sample:
        data1 = read_data(gold_data_path)
        random.seed(sample)
        random.shuffle(data1)
        gold_data = data1[:20]
    else:
        gold_data = read_data(gold_data_path)

    gold_answers = extract_answers(gold_data)
    pred_data = load_data(pred_data_path)

    if cot:
        pred_answers = extract_cot_with_example_pred_answers(pred_data)
        # pred_answers is a tuple (refined_preds, all_preds),
        # but for standard CoT accuracy you might only need refined_preds
        pred_answers = pred_answers[0]
    else:
        pred_answers = extract_pred_answers(pred_data)

    correct = 0
    for i in range(len(gold_answers)):
        if gold_answers[i] == pred_answers[i]:
            correct += 1
            gold_data[i]['prediction'] = pred_data[i]
            gold_data[i]['correct'] = True
        else:
            gold_data[i]['prediction'] = pred_data[i]
            gold_data[i]['correct'] = False

    if verbose:
        print("Accuracy: ", correct / len(gold_answers))
        print("Total: ", len(gold_answers))
        print("Correct: ", correct)
        stat = {
            "correct": correct,
            "total": len(gold_answers),
            "accuracy": correct / len(gold_answers)
        }
        save_data(stat, 'results.txt')

    if save_path:
        result_path = save_path.split('.')[0] + "_results.txt"
        save_data(gold_data, save_path)
        save_data(stat, result_path)

def compute_cot_with_example_accuracy(gold_data_path, pred_data_path, save_path=None, verbose=True, sample=None):
    if sample:
        pdb.set_trace()
        data1 = read_data(gold_data_path)
        random.seed(sample)
        random.shuffle(data1)
        gold_data = data1[:20]
    else:
        gold_data = read_data(gold_data_path)

    gold_answers = extract_answers(gold_data)
    pred_data = load_data(pred_data_path)
    pred_answers, pred_all = extract_cot_with_example_pred_answers(pred_data)
    
    correct = 0
    for i in range(len(gold_answers)):
        if gold_answers[i] == pred_answers[i]:
            correct += 1
            gold_data[i]['prediction'] = pred_data[i]
            gold_data[i]['correct'] = True
            gold_data[i]['pred_answer'] = pred_answers[i]
            gold_data[i]['pred_only'] = pred_all[i]
        else:
            gold_data[i]['prediction'] = pred_data[i]
            gold_data[i]['correct'] = False
            gold_data[i]['pred_answer'] = pred_answers[i]
            gold_data[i]['pred_only'] = pred_all[i]

    if verbose:
        print("Accuracy: ", correct/len(gold_answers))
        print("Total: ", len(gold_answers))
        print("Correct: ", correct)

    stat = {
        "correct": correct,
        "total": len(gold_answers),
        "accuracy": correct/len(gold_answers)
    }
    if save_path:
        result_path = save_path.split('.')[0] + "_results.txt"
        save_data(gold_data, save_path)
        save_data(stat, result_path)

###############################################################################
# Qwen-specific classes and inference
###############################################################################
class MyQwen:
    """
    Simple Qwen wrapper that replicates the interface from your Llama-based code.
    """
    def __init__(self, model_name_or_path, max_seq_len=2048, device="cuda"):
        print(f"Loading Qwen model from {model_name_or_path} with torch.bfloat16 ...")
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        self.max_seq_len = max_seq_len

    def inference(self, prompts, max_gen_len=300):
        """
        Run inference on a list of prompts. Return the generated strings.
        """
        results = []
        for prompt_text in prompts:
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_gen_len,
                    do_sample=True,       # or False, depending on your use case
                    temperature=0.7,      # adjust as needed
                    top_p=0.9,            # adjust as needed
                )
            generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # If you need to isolate just the newly generated portion, you'd do:
            # generated = self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            results.append(generated)
        return results

###############################################################################
# Data Cleaning and Inference Routines
###############################################################################
def clean_cot_pred_answers(data, save_path=None):
    clean_data = []
    for each in data:
        each_dict = {}
        each_dict['original'] = each
        each_dict['cleaned'] = each.split('\nQ:')[0].strip() + " Therefore, the answer (arabic numerals) is "
        clean_data.append(each_dict)
    if save_path:
        save_data(clean_data, save_path)
    return clean_data

def extract_cot_pred_answers(generator, max_gen_len=300, data_path="results/batch/test_cot_results.jsonl", save_path=None):
    data = load_data(data_path)
    clean_data_list = clean_cot_pred_answers(data)
    results = []
    for i in tqdm(range(0, len(clean_data_list), 6)):
        batch = clean_data_list[i:i+6]
        prompts = [item['cleaned'] for item in batch]
        preds = generator.inference(prompts, max_gen_len=max_gen_len)
        results.extend(preds)
    
    if save_path:
        save_data(results, save_path)

def run_inference(generator, data, prompt, type="cot", batch_size=6, max_gen_len=300, save_path=None, few_shot=False):
    results = []
    data_copy = copy.deepcopy(data)
    for i in tqdm(range(0, len(data_copy), batch_size)):
        batch = data_copy[i:i+batch_size]

        for j in range(len(batch)):
            # Remove extra spaces in the question
            batch[j]['question'] = ' '.join(batch[j]['question'].split())
            if few_shot:
                if type == "equation_only":
                    batch[j]['question'] = prompt + 'Question: ' + batch[j]['question'] + "\n\n"
                else:
                    batch[j]['question'] = prompt + 'Question: ' + batch[j]['question'] + "\nLet's think step by step\n"
            else:
                # single-shot or zero-shot
                batch[j]['question'] = 'Q: ' + batch[j]['question'] + " A: " + prompt

        final_prompts = [each['question'] for each in batch]
        preds = generator.inference(final_prompts, max_gen_len=max_gen_len)
        results.extend(preds)

    if save_path:
        # Save as a JSON list of strings, consistent with your approach
        save_data(results, save_path)

###############################################################################
# Main
###############################################################################
def main(
    model_name_or_path: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    prompt: str = "prompts/example_prompt.txt",
    few_shot: bool = False,
    max_seq_len: int = 2048,
    max_gen_len: int = 500,
    max_batch_size: int = 6,
    data_path: str = "data/test.jsonl",
    results_dir: str = "results/gsm8k_inference/",
    device: str = "cuda"
):
    # 1) Load data
    data = read_data(data_path)
    # 2) Load prompt
    if few_shot:
        if '.txt' not in prompt:
            raise SystemExit("Provide path to your prompt file (.txt) containing few-shot demonstration!")
        else:
            # Minimal read: treat the text file as a single string
            with open(prompt, 'r', encoding='utf-8') as f:
                prompt_lines = f.read()
            clean_prompt = prompt_lines + '\n\n'
            prompt_type = os.path.basename(prompt).replace(".txt", "")
    else:
        prompt_type = prompt.replace(" ", "_")  # or something more robust
        clean_prompt = prompt

    # 3) Create result directories
    if not os.path.exists(results_dir):
        print("Creating directory:", results_dir)
        os.makedirs(results_dir)
    # Subdirectory for the given prompt_type
    sub_dir = os.path.join(results_dir, prompt_type)
    if not os.path.exists(sub_dir):
        print("Creating directory:", sub_dir)
        os.makedirs(sub_dir)

    # 4) Paths to store intermediate/final results
    inference_save_path = os.path.join(sub_dir, "results.jsonl")
    extract_pred_path = os.path.join(sub_dir, "clean_results.jsonl")
    final_save_path = os.path.join(sub_dir, "final.json")
    print("Saving to:", inference_save_path)

    # 5) Load Qwen model
    generator = MyQwen(
        model_name_or_path=model_name_or_path,
        max_seq_len=max_seq_len,
        device=device
    )

    # 6) Run inference
    run_inference(
        generator,
        data,
        batch_size=max_batch_size,
        prompt=clean_prompt,
        type=prompt_type,
        few_shot=few_shot,
        save_path=inference_save_path,
        max_gen_len=max_gen_len
    )

    # 7) Compute accuracy
    if few_shot:
        # This uses the CoT extraction logic
        compute_cot_with_example_accuracy(
            gold_data_path=data_path,
            pred_data_path=inference_save_path,
            save_path=final_save_path
        )
    else:
        # Zero-shot or single-shot
        extract_cot_pred_answers(
            generator,
            data_path=inference_save_path,
            save_path=extract_pred_path,
            max_gen_len=100
        )
        compute_accuracy(
            gold_data_path=data_path,
            pred_data_path=extract_pred_path,
            save_path=final_save_path
        )

if __name__ == "__main__":
    fire.Fire(main)