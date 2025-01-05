import json
import pdb
from tqdm import tqdm
import random
import re
import copy
import torch
import os
import sys
from scripts import utils
from typing import List
import fire

# Removed: from scripts.Llama import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import prompts


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
        each_pred = re.sub(r'[a-zA-Z%$=\-]', ' ', each_pred)  # remove alphabets
        each_pred = each_pred.replace(",", "")                # remove commas
        each_pred = ' '.join(each_pred.split())               # remove multiple spaces
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

def compute_accuracy(gold_data_path, pred_data_path, cot=False, save_path=None, verbose=True, sample=None):
    if sample:
        data1 = utils.read_data(gold_data_path)
        random.seed(sample)
        random.shuffle(data1)
        gold_data = data1[:20]
    else:
        gold_data = read_data(gold_data_path)

    gold_answers = extract_answers(gold_data)
    pred_data = load_data(pred_data_path)
    if cot:
        pred_answers = extract_cot_with_example_pred_answers(pred_data)
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
        print("Accuracy: ", correct/len(gold_answers))
        print("Total: ", len(gold_answers))
        print("Correct: ", correct)
        stat = {
            "correct": correct,
            "total": len(gold_answers),
            "accuracy": correct/len(gold_answers)
        }
        save_data(stat, 'results.txt')

    if save_path:
        result_path = save_path.split('.')[0] + "_results.txt"
        save_data(gold_data, save_path)
        save_data(stat, result_path)

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
        pred4 = pred3.replace(",", "") 
        pred5 = ' '.join(pred4.split()) 
        try:
            pred6 = pred5.split()[-1]
            pred7 = int(float(pred6))
        except:
            pred7 = 234234
        refined_preds.append(pred7)
    return refined_preds, all_preds

def compute_cot_with_example_accuracy(gold_data_path, pred_data_path, save_path=None, verbose=True, sample=None):
    if sample:
        pdb.set_trace()
        data1 = utils.read_data(gold_data_path)
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


def extract_cot_pred_answers(generator, max_gen_len=300, data_path="results/batch/test_cot_results.jsonl", save_path=None):
    data = load_data(data_path)
    clean_data = clean_cot_pred_answers(data)
    # Inference in batches of 6, as before
    results = []
    for i in tqdm(range(0, len(clean_data), 6)):
        batch = clean_data[i:i+6]
        prompts = [each['cleaned'] for each in batch]
        preds = generator.inference(prompts, max_gen_len=max_gen_len)
        results.extend(preds)
    
    if save_path:
        save_data(results, save_path)

def clean_cot_pred_answers(data, save_path=None):
    clean_data = []
    for each in data:
        each_dict = {}
        each_dict['original'] = each
        each_dict['cleaned'] = each.split('\nQ:')[0].strip() + " Therefore, the answer (arabic numerals) is "
        clean_data.append(each_dict)
    if save_path:
        utils.save_data(clean_data, save_path)
    return clean_data


def run_inference(generator, data, prompt, type="cot", batch_size=6, max_gen_len=300, save_path=None, few_shot=False):
    results = []
    data_copy = copy.deepcopy(data)
    for i in tqdm(range(0, len(data_copy), batch_size)):
        batch = data_copy[i:i+batch_size]

        for j in range(len(batch)):
            # Clean up multiple spaces
            batch[j]['question'] = ' '.join(batch[j]['question'].split())
            if few_shot:
                if type == "equation_only":
                    batch[j]['question'] = prompt + 'Question: ' + batch[j]['question'] + "\n\n"
                else:
                    batch[j]['question'] = prompt + 'Question: ' + batch[j]['question'] + "\nLet's think step by step\n"
            else:
                batch[j]['question'] = 'Q: ' + batch[j]['question'] + " A: " + prompt
        
        final_prompts = [each['question'] for each in batch]

        preds = generator.inference(final_prompts, max_gen_len=max_gen_len)
        results.extend(preds)

    if save_path:
        save_data(results, save_path)


#
# REPLACE the original MyLlama(Llama) with a Hugging Face wrapper
#
class MyHFModel:
    """Simple Hugging Face model wrapper to replicate the 'generator.inference()' interface."""
    def __init__(
        self,
        model_path,
        tokenizer_path,
        max_seq_len: int = 500,
        max_gen_len: int = 400,
        max_batch_size: int = 6,
        model_parallel_size=None
    ):
        # Just use model_path for both model & tokenizer (or separately if needed).
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16  # or "auto", etc.
        )
        self.model.eval()
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.max_batch_size = max_batch_size

    def inference(self, prompts: List[str], max_gen_len=300) -> List[str]:
        """Generate text for a list of prompt strings."""
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_gen_len,
                    # temperature, top_p, etc., could be added if you want
                )
            gen = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(gen)
        return results


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    prompt: str,
    few_shot: bool = True,
    max_seq_len: int = 4500,
    max_gen_len: int = 500,
    max_batch_size: int = 6,
    model_parallel_size=None,
    data_path="data/test.jsonl",
    results_dir="results/gsm8k_inference/"
):
    batch_size = max_batch_size

    # load data
    data = read_data(data_path)

    # load prompt
    if few_shot:
        if '.txt' not in prompt:
            raise SystemExit("Provide path to your prompt file (.txt) containing few-shot demonstration!")
        else:
            prompt1 = utils.read_file(prompt)
            clean_prompt = '\n'.join(prompt1) + '\n\n'
            prompt_type = prompt.split("/")[-1].replace(".txt", "")
    else:
        prompt_type = prompt.split()[0] + "_" + prompt.split()[-1]
        clean_prompt = prompt

    # Create directories for saving results
    if not os.path.exists(results_dir):
        print("Creating directory: ", results_dir)
        os.makedirs(results_dir)
    if not os.path.exists(f"{results_dir}/{prompt_type}"):
        print("Creating directory: ", f"{results_dir}/{prompt_type}")
        os.makedirs(f"{results_dir}/{prompt_type}")

    inference_save_path = f"{results_dir}/{prompt_type}/results.jsonl"
    extract_pred_path = f"{results_dir}/{prompt_type}/clean_results.jsonl"
    final_save_path = f"{results_dir}/{prompt_type}/final.json"
    print("Saving intermediate results to:", inference_save_path)

    #
    # Create our HF generator
    #
    generator = MyHFModel(
        model_path=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_gen_len=max_gen_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size
    )

    # Run inference
    run_inference(
        generator,
        data,
        batch_size=batch_size,
        prompt=clean_prompt,
        type=prompt_type,
        few_shot=few_shot,
        save_path=inference_save_path,
        max_gen_len=max_gen_len
    )

    # Compute accuracy
    if few_shot:
        compute_cot_with_example_accuracy(
            gold_data_path=data_path,
            pred_data_path=inference_save_path,
            save_path=final_save_path
        )
    else:
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
    # torchrun --nproc_per_node 1 gsm8k_inference.py --ckpt_dir ../../../downloads/huggingface/models/llama2-7b/ --tokenizer_path ../../../downloads/huggingface/models/llama2-7b/tokenizer.model --prompt "Let's think step by step" --few_shot False

        
    