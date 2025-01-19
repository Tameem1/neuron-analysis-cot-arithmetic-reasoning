import json
import csv

def final_json_to_csv(json_path: str, csv_path: str):
    """
    Reads the file at `json_path`, which is expected to be a list of lists of objects,
    e.g. [ [ { 'question':..., 'answer':..., 'prediction':..., 'correct':...}, ...], 
           [ { 'question':..., 'answer':..., 'prediction':..., 'correct':...}, ...] ].

    We flatten this list of lists, and then write the result to a CSV file at `csv_path`
    with columns: question, answer, prediction, correct.
    """
    # 1. Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # data is a list of lists

    # 2. Flatten the list of lists into a single list of objects
    #    Example: [[obj1, obj2], [obj3, obj4]] --> [obj1, obj2, obj3, obj4]
    flattened_data = []
    for sublist in data:
        flattened_data.extend(sublist)

    # 3. Write to CSV
    fieldnames = ['question', 'answer', 'prediction', 'correct']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in flattened_data:
            writer.writerow({
                'question':   item.get('question', ''),
                'answer':     item.get('answer', ''),
                'prediction': item.get('prediction', ''),
                'correct':    item.get('correct', '')
            })

if __name__ == "__main__":
    input_json_file = "/Users/tameem/Documents/GitHub/neuron-analysis-cot-arithmetic-reasoning/results/chunks_direct/final.json"       # Path to your JSON file (list of lists)
    output_csv_file = "/Users/tameem/Documents/GitHub/neuron-analysis-cot-arithmetic-reasoning/results/final_output.csv" # Where to save the CSV
    final_json_to_csv(input_json_file, output_csv_file)
    print(f"CSV file created: {output_csv_file}")