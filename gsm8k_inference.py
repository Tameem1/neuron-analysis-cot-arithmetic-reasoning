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

from transformers import AutoTokenizer, AutoModelForCausalLM
from data import prompts  # Assuming this is your local import
from scripts import utils  # Assuming this is your local import


def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_data(file_path):
    return json.load(open(file_path, "r", encoding="utf-8"))


def read_data(file_path):
    raw_data = []
    with open(file_path, "r", encoding='utf-8') as fin:
        for line in fin:
            raw_data.append(json.loads(line))
    return raw_data


def extract_answers(data):
    answers = []
    for each in data:
        # Example parsing: "answer": "... \n#### 91"
        # We split on '\n#### ' and convert the number to int
        each_answer = int(each['answer'].split('\n#### ')[1].replace(",", ""))
        answers.append(each_answer)
    return answers


def extract_pred_answers(data, verbose=False):
    """
    Extract integer predictions from text completions.
    This is project-specific logic: we look for 'answer (arabic numerals) is ',
    then parse the trailing integer.
    """
    preds = []
    for each in data:
        try:
            # 1) Split at 'answer (arabic numerals) is '
            each_pred = each.split('answer (arabic numerals) is ')[1].split('.\n')[0]
        except:
            # If we fail, default to "0"
            each_pred = "0"

        # 2) Remove alphabets/symbols, keep numeric
        each_pred = re.sub(r'[a-zA-Z%$=\-]', ' ', each_pred)
        each_pred = each_pred.replace(",", "")
        each_pred = ' '.join(each_pred.split())  # remove multiple spaces

        # 3) Usually the last token is the numeric answer
        tokens = each_pred.split()
        if tokens:
            each_pred = tokens[-1]
        else:
            each_pred = "0"

        # 4) Convert to integer
        try:
            each_pred = int(float(each_pred))
        except:
            each_pred = 0

        if verbose:
            print("Full text:", each)
            print("Parsed pred:", each_pred)
            print("----")
        preds.append(each_pred)

    return preds


def compute_accuracy(gold_data_path, pred_data_path, cot=False, save_path=None, verbose=True, sample=None):
    """
    Compare gold answers to predicted answers (that were previously saved).
    """
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
        accuracy_val = correct / len(gold_answers)
        print("Accuracy: ", accuracy_val)
        print("Total: ", len(gold_answers))
        print("Correct: ", correct)
        stat = {
            "correct": correct,
            "total": len(gold_answers),
            "accuracy": accuracy_val
        }
        save_data(stat, 'results.txt')

    if save_path:
        result_path = save_path.split('.')[0] + "_results.txt"
        save_data(gold_data, save_path)
        if verbose:
            save_data(stat, result_path)


def extract_cot_with_example_pred_answers(preds):
    """
    Example function for chain-of-thought output parsing.
    """
    refined_preds = []
    all_preds = []
    for pred in preds:
        try:
            # The code below is project-specific. 
            # It's taking "The answer is " and some index [9], etc.
            # Adjust as needed to parse your CoT format.
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
    """
    Similar to compute_accuracy but for chain-of-thought results, 
    storing "pred_answer" and "pred_only".
    """
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
        accuracy_val = correct / len(gold_answers)
        print("Accuracy: ", accuracy_val)
        print("Total: ", len(gold_answers))
        print("Correct: ", correct)
    stat = {
        "correct": correct,
        "total": len(gold_answers),
        "accuracy": correct / len(gold_answers)
    }
    if save_path:
        result_path = save_path.split('.')[0] + "_results.txt"
        save_data(gold_data, save_path)
        save_data(stat, result_path)


def extract_cot_pred_answers(generator, max_gen_len=300, data_path="results/batch/test_cot_results.jsonl", save_path=None):
    """
    Example function that re-infers from partially cleaned data, in case your pipeline needs it.
    """
    data = load_data(data_path)
    clean_data = clean_cot_pred_answers(data)
    results = []
    # Now do inference in steps of 6
    for i in tqdm(range(0, len(clean_data), 6)):
        batch = clean_data[i : i + 6]
        prompts = [each['cleaned'] for each in batch]
        preds = generator.inference(prompts, max_gen_len=max_gen_len)
        results.extend(preds)
    
    if save_path:
        save_data(results, save_path)


def clean_cot_pred_answers(data, save_path=None):
    """
    Example function for reformatting chain-of-thought data into a shorter prompt,
    if your pipeline needs it.
    """
    clean_data = []
    for each in data:
        each_dict = {}
        each_dict['original'] = each
        # Basic example: strip everything after the first '\nQ:' and add "Therefore..." at the end
        each_dict['cleaned'] = each.split('\nQ:')[0].strip() + " Therefore, the answer (arabic numerals) is "
        clean_data.append(each_dict)
    if save_path:
        utils.save_data(clean_data, save_path)
    return clean_data


#
# NEW MyHFModel with truncation, batch generation, and advanced gen parameters
#
class MyHFModel:
    """Simple Hugging Face model wrapper to replicate the 'generator.inference()' interface,
       with truncation, batching, and extra generation hyperparameters.
    """
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_seq_len: int = 500,
        max_gen_len: int = 400,
        max_batch_size: int = 6,
        model_parallel_size=None
    ):
        # If tokenizer is in a separate directory, use tokenizer_path explicitly:
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Otherwise, if both are in model_path:
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

    def inference(
        self,
        prompts: List[str],
        max_gen_len: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        """
        Generate text for a list of prompt strings in mini-batches, 
        with truncation, optional generation parameters, etc.
        """
        results = []
        for i in range(0, len(prompts), self.max_batch_size):
            batch_prompts = prompts[i : i + self.max_batch_size]

            # Tokenize with truncation and padding
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                max_length=self.max_seq_len,  # Enforce input truncation
                truncation=True,
                padding=True
            ).to(self.model.device)

            with torch.no_grad():
                # Generate up to max_gen_len new tokens
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    # e.g. do_sample=True if you want sampling
                )

            # Decode each sample in the batch
            for out_seq in outputs:
                gen_text = self.tokenizer.decode(out_seq, skip_special_tokens=True)
                results.append(gen_text)

        return results


#
# Updated run_inference (single pass)
#
def run_inference(
    generator: MyHFModel,
    data,
    prompt: str,
    type="cot",
    max_gen_len=300,
    save_path=None,
    few_shot=False
):
    """
    Build all final prompts at once, then let 'generator.inference' 
    handle the actual batch generation internally.
    """
    data_copy = copy.deepcopy(data)
    final_prompts = []

    # Build final prompts
    for item in data_copy:
        # Clean up multiple spaces in question
        qtext = ' '.join(item['question'].split())

        if few_shot:
            if type == "equation_only":
                text_prompt = prompt + f"Question: {qtext}\n\n"
            else:
                text_prompt = prompt + f"Question: {qtext}\nLet's think step by step\n"
        else:
            text_prompt = f"Q: {qtext} A: {prompt}"

        final_prompts.append(text_prompt)

    # Inference in one shot
    preds = generator.inference(
        final_prompts,
        max_gen_len=max_gen_len,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0
    )

    # Save predictions if needed
    if save_path:
        save_data(preds, save_path)


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
    # 1) Load data
    data = read_data(data_path)
    data = data[:10]
    # 2) Build or load prompt
    if few_shot:
        # The user must pass a .txt with few-shot examples
        if '.txt' not in prompt:
            raise SystemExit("Provide path to your prompt file (.txt) containing few-shot demonstration!")
        else:
            prompt_content = utils.read_file(prompt)
            clean_prompt = '\n'.join(prompt_content) + '\n\n'
            prompt_type = os.path.basename(prompt).replace(".txt", "")
    else:
        # Single-shot or no-shot scenario
        prompt_type = prompt.split()[0] + "_" + prompt.split()[-1]
        clean_prompt = prompt

    # 3) Create directories to save results
    if not os.path.exists(results_dir):
        print("Creating directory:", results_dir)
        os.makedirs(results_dir)

    out_dir = f"{results_dir}/{prompt_type}"
    if not os.path.exists(out_dir):
        print("Creating directory:", out_dir)
        os.makedirs(out_dir)

    inference_save_path = f"{out_dir}/results.jsonl"
    extract_pred_path = f"{out_dir}/clean_results.jsonl"
    final_save_path = f"{out_dir}/final.json"
    print("Saving intermediate results to:", inference_save_path)

    # 4) Initialize our HF generator with advanced parameters
    generator = MyHFModel(
        model_path=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_gen_len=max_gen_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size
    )

    # 5) Run inference
    run_inference(
        generator=generator,
        data=data,
        prompt=clean_prompt,
        type=prompt_type,
        max_gen_len=max_gen_len,
        save_path=inference_save_path,
        few_shot=few_shot
    )

    # 6) Compute accuracy
    if few_shot:
        # If you have chain-of-thought or multi-answer parsing:
        compute_cot_with_example_accuracy(
            gold_data_path=data_path,
            pred_data_path=inference_save_path,
            save_path=final_save_path
        )
    else:
        # If standard (no CoT):
        extract_cot_pred_answers(
            generator=generator,
            data_path=inference_save_path,
            save_path=extract_pred_path,
            max_gen_len=500
        )
        compute_accuracy(
            gold_data_path=data_path,
            pred_data_path=extract_pred_path,
            save_path=final_save_path
        )


if __name__ == "__main__":
    fire.Fire(main)
    # Example usage:
    # torchrun --nproc_per_node 1 gsm8k_inference.py \
    #   --ckpt_dir path/to/huggingface/models/llama2-7b/ \
    #   --tokenizer_path path/to/huggingface/models/llama2-7b/tokenizer.model \
    #   --prompt "Let's think step by step" \
    #   --few_shot False