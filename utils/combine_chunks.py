import os
import json

def combine_json_files(root_folder: str):
    """
    Walk through all subfolders of `root_folder`. 
    - If a subfolder contains the files:
        * clean_results.jsonl
        * final.json
        * results.jsonl
      we read them and accumulate their data.
    - Otherwise, we notify (print) which files are missing in that folder.

    Finally, we output three combined files in the `root_folder`:

        - clean_results.jsonl  (concatenated lines from all clean_results.jsonl)
        - final.json           (an array of all final.json contents)
        - results.jsonl        (concatenated lines from all results.jsonl)
    """
    
    # Lists to accumulate lines from .jsonl files
    all_clean_results_lines = []
    all_results_lines = []
    
    # List to accumulate objects from final.json
    all_final_objects = []
    
    # Required files
    required_files = {"clean_results.jsonl", "final.json", "results.jsonl"}
    
    # Traverse the directory tree
    for current_path, dirs, files in os.walk(root_folder):
        # Convert files list to a set for easy membership checking
        file_set = set(files)
        
        # Check if the current directory contains the three required files
        if required_files.issubset(file_set):
            # Paths to the three files
            clean_results_path = os.path.join(current_path, "clean_results.jsonl")
            final_json_path    = os.path.join(current_path, "final.json")
            results_jsonl_path = os.path.join(current_path, "results.jsonl")
            
            # Read clean_results.jsonl (line by line)
            with open(clean_results_path, 'r', encoding='utf-8') as f:
                all_clean_results_lines.extend(f.readlines())
            
            # Read results.jsonl (line by line)
            with open(results_jsonl_path, 'r', encoding='utf-8') as f:
                all_results_lines.extend(f.readlines())
            
            # Read final.json (assumed to be a single JSON object or an array)
            with open(final_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Append the JSON data to our list
                all_final_objects.append(data)
                
        else:
            # Identify which required files are missing
            missing = required_files - file_set
            print(f"Folder '{current_path}' is missing the following required file(s): {missing}")
    
    # --- Write out the combined files to the root_folder ---
    combined_clean_results_path = os.path.join(root_folder, "clean_results.jsonl")
    combined_results_path       = os.path.join(root_folder, "results.jsonl")
    combined_final_path         = os.path.join(root_folder, "final.json")
    
    # 1. Write combined clean_results.jsonl
    with open(combined_clean_results_path, 'w', encoding='utf-8') as f:
        f.writelines(all_clean_results_lines)
    
    # 2. Write combined results.jsonl
    with open(combined_results_path, 'w', encoding='utf-8') as f:
        f.writelines(all_results_lines)
    
    # 3. Write combined final.json
    #    Here we write the list of final.json contents as a JSON array.
    with open(combined_final_path, 'w', encoding='utf-8') as f:
        json.dump(all_final_objects, f, indent=2)

if __name__ == "__main__":
    # Example usage:
    path_to_root_folder = "/Users/tameem/Documents/GitHub/neuron-analysis-cot-arithmetic-reasoning/results/chunks_direct/"
    combine_json_files(path_to_root_folder)