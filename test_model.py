import json
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_few_shot_prompt(prompt_path: str) -> str:
    """
    Load the few-shot prompt from a text file.

    Args:
        prompt_path (str): Path to the few-shot prompt file.

    Returns:
        str: The few-shot prompt as a single string.
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        few_shot = f.read()
    # Ensure proper formatting: add two newlines at the end
    if not few_shot.endswith('\n\n'):
        few_shot += '\n\n'
    return few_shot

def load_test_questions(test_path: str, num_samples: int = 4) -> List[str]:
    """
    Load a specified number of questions from the test.jsonl file.

    Args:
        test_path (str): Path to the test.jsonl file.
        num_samples (int, optional): Number of questions to load. Defaults to 4.

    Returns:
        List[str]: A list of question strings.
    """
    questions = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
            if len(questions) >= num_samples:
                break
    return questions

def construct_prompt(few_shot: str, question: str, prompt_type: str = "equation_only") -> str:
    """
    Construct the prompt by combining few-shot examples with the new question.

    Args:
        few_shot (str): The few-shot prompt.
        question (str): The new question to be answered.
        prompt_type (str, optional): Type of prompt. Defaults to "equation_only".

    Returns:
        str: The complete prompt to be sent to the model.
    """
    if prompt_type == "equation_only":
        prompt = f"{few_shot}Question: {question}\nAnswer:"
    else:
        prompt = f"{few_shot}Question: {question}\nLet's think step by step\nAnswer:"
    return prompt

def perform_inference(model, tokenizer, prompts: List[str], max_gen_len: int = 100) -> List[str]:
    """
    Generate predictions for a list of prompts using the model.

    Args:
        model: The language model.
        tokenizer: The tokenizer corresponding to the model.
        prompts (List[str]): List of prompt strings.
        max_gen_len (int, optional): Maximum number of tokens to generate. Defaults to 100.

    Returns:
        List[str]: List of generated prediction strings.
    """
    # Tokenize the prompts with left padding
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    # Assign pad_token to eos_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token not found. Setting pad_token to eos_token.")

    # Move inputs to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.to(device)

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

    predictions = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    return predictions

def main():
    # Hardcoded paths
    few_shot_path = "data/prompts/equation_only.txt"  # Path to few-shot prompt
    test_path = "data/test.jsonl"                         # Path to test data

    # Model configuration
    model_name = "~/models/qwen2.5-math-1.5B-instruct"                                   # Hugging Face model name; choose as desired
    tokenizer_name = "~/models/qwen2.5-math-1.5B-instruct/"                               # Corresponding tokenizer

    # Number of samples to test
    num_samples = 4

    # Prompt type
    prompt_type = "equation_only"                         # Can be "equation_only" or other types as defined

    # Maximum tokens to generate
    max_gen_len = 700

    # Load the few-shot prompt
    few_shot = load_few_shot_prompt(few_shot_path)
    print("Few-shot prompt loaded.")

    # Load a small set of test questions
    test_questions = load_test_questions(test_path, num_samples=num_samples)
    print(f"Loaded {len(test_questions)} test questions.")

    # Initialize the tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Assign pad_token to eos_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token not found. Setting pad_token to eos_token.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )
    model.eval()
    print("Model and tokenizer loaded.")

    # Construct prompts for each question
    prompts = [construct_prompt(few_shot, q, prompt_type=prompt_type) for q in test_questions]
    for idx, prompt in enumerate(prompts):
        print(f"\n--- Prompt {idx+1} ---\n{prompt}\n")

    # Perform inference
    print("Starting inference...")
    predictions = perform_inference(model, tokenizer, prompts, max_gen_len=max_gen_len)
    print("Inference completed.\n")

    # Display and save predictions
    print("--- Predictions ---")
    for idx, pred in enumerate(predictions):
        print(f"\nPrediction {idx+1}:\n{pred}\n")
        # Optionally, save each prediction to a separate file for detailed analysis
        with open(f"prediction_{idx+1}.txt", "w", encoding="utf-8") as f:
            f.write(pred)

if __name__ == "__main__":
    main()