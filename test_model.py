import json
import os
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_few_shot_prompt(prompt_path: str) -> str:
    with open(prompt_path, 'r', encoding='utf-8') as f:
        few_shot = f.read()
    if not few_shot.endswith('\n\n'):
        few_shot += '\n\n'
    return few_shot

def load_test_questions(test_path: str, num_samples: int = 4) -> List[str]:
    questions = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
            if len(questions) >= num_samples:
                break
    return questions

def construct_prompt(few_shot: str, question: str, prompt_type: str = "equation_only") -> str:
    if prompt_type == "equation_only":
        prompt = f"{few_shot}Question: {question}\n\n"
    else:
        prompt = f"{few_shot}Question: {question}\nLet's think step by step\n"
    return prompt

def perform_inference(model, tokenizer, prompts: List[str], max_gen_len: int = 100) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return predictions

def main():
    # Hardcoded paths
    few_shot_path = "data/prompts/equation_only.txt"         # Path to few-shot prompt
    test_path = "data/test.jsonl"                            # Path to test data
    model_path = "/export/home/tkabbani/models/qwen2.5-math-1.5B-instruct"       # Path to your model directory
    tokenizer_path = "/export/home/tkabbani/models/qwen2.5-math-1.5B-instruct/tokenizer.json"  # Path to your tokenizer

    # Number of samples to test
    num_samples = 4

    # Prompt type
    prompt_type = "equation_only"

    # Maximum tokens to generate
    max_gen_len = 100

    # Load the few-shot prompt
    few_shot = load_few_shot_prompt(few_shot_path)
    print("Few-shot prompt loaded.")

    # Load a small set of test questions
    test_questions = load_test_questions(test_path, num_samples=num_samples)
    print(f"Loaded {len(test_questions)} test questions.")

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16  # Adjust dtype if necessary
    )
    model.eval()
    print("Model and tokenizer loaded.")

    # Construct prompts for each question
    prompts = [construct_prompt(few_shot, q, prompt_type=prompt_type) for q in test_questions]
    for idx, prompt in enumerate(prompts):
        print(f"\n--- Prompt {idx+1} ---\n{prompt}\n")

    # Perform inference
    predictions = perform_inference(model, tokenizer, prompts, max_gen_len=max_gen_len)
    print("\n--- Predictions ---")
    for idx, pred in enumerate(predictions):
        print(f"\nPrediction {idx+1}:\n{pred}\n")
        # Optionally, save each prediction to a separate file for detailed analysis
        with open(f"prediction_{idx+1}.txt", "w", encoding="utf-8") as f:
            f.write(pred)
        print(f"Prediction {idx+1} saved to prediction_{idx+1}.txt")

if __name__ == "__main__":
    main()