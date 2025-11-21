import os
import json
import copy
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report

# ==========================================
# Configuration
# ==========================================

API_KEY = "your_api_key_here"  # os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# File Paths
DATA_DIR = "./data/"
OUTPUT_DIR = "./results/"
TEST_FILE = os.path.join(DATA_DIR, "D4_screening_conversation_testing_translated.json")
FEW_SHOT_FILE = os.path.join(DATA_DIR, "D4_few_shot.json")
GROUND_TRUTH_FILE = os.path.join(OUTPUT_DIR, "D4_grdtruth.xlsx")

# ==========================================
# Helper Functions
# ==========================================

def get_llm_response(prompt, system_role="", model="gpt-4"):
    """Interacts with OpenAI API."""
    try:
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ]
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        return "{}"

def binary_transform(data_list):
    """Converts multi-class risk levels (0-3) to binary (0 vs 1)."""
    transformed = copy.deepcopy(data_list)
    for item in transformed:
        item['Depression_Risk'] = 1 if item['Depression_Risk'] >= 1 else 0
        item['Suicide_Risk'] = 1 if item['Suicide_Risk'] >= 1 else 0
    return transformed

def build_fewshot_prompt(train_data):
    """Constructs the few-shot prompt string."""
    examples = "Here are some examples you can refer to: \n\n"
    
    for i, sample in enumerate(train_data):
        # Format the target output as JSON string
        output_json = json.dumps({
            "Depression_Risk": sample['Depression_Risk'],
            "Suicide_Risk": sample['Suicide_Risk'],
            "Summary": sample['Summary']
        })
        
        examples += f"<start_of_Example_{i+1}>:\n"
        examples += sample['prompt']
        examples += "\n" + output_json
        examples += f"\n<end_of_Example_{i+1}>\n\n"

    examples += "Now please do the following task: \n\n"
    return examples

def run_inference(data, prompt_prefix=""):
    """Iterates through data and gets predictions."""
    results = []
    for sample in tqdm(data):
        query = prompt_prefix + sample['prompt']
        response_text = get_llm_response(query)
        
        try:
            parsed_json = json.loads(response_text)
            results.append(parsed_json)
        except json.JSONDecodeError:
            # Fallback for malformed JSON
            results.append({"Depression_Risk": 0, "Suicide_Risk": 0, "Summary": "Error parsing JSON"})
            
    return results

def evaluate_performance(y_true, y_pred, label_name):
    """Prints classification report."""
    print(f"--- Evaluation: {label_name} ---")
    print(classification_report(y_true, y_pred, zero_division=0))

# ==========================================
# Main Execution
# ==========================================

def main():
    # 1. Load and Preprocess Data
    if not os.path.exists(TEST_FILE) or not os.path.exists(FEW_SHOT_FILE):
        print("Data files not found.")
        return

    with open(TEST_FILE, 'r') as f:
        raw_test_data = json.load(f)
    
    with open(FEW_SHOT_FILE, 'r') as f:
        raw_train_data = json.load(f)

    # Convert to binary classification task
    test_data = binary_transform(raw_test_data)
    train_data = binary_transform(raw_train_data)

    # 2. Generate Ground Truth DataFrame
    gt_data = {
        'Depression_Risk': [x['Depression_Risk'] for x in test_data],
        'Suicide_Risk': [x['Suicide_Risk'] for x in test_data],
        'Summary': [x['Summary'] for x in test_data]
    }
    df_ground_truth = pd.DataFrame(gt_data)
    df_ground_truth.to_excel(GROUND_TRUTH_FILE, index=False)

    # 3. Zero-shot Inference
    print("Running Zero-shot Inference...")
    results_zeroshot = run_inference(test_data)
    
    df_zeroshot = pd.DataFrame(results_zeroshot)
    df_zeroshot.to_excel(os.path.join(OUTPUT_DIR, "gpt4_d4_zeroshot.xlsx"), index=False)

    # 4. Few-shot Inference
    print("Running Few-shot Inference...")
    fewshot_prompt = build_fewshot_prompt(train_data)
    results_fewshot = run_inference(test_data, prompt_prefix=fewshot_prompt)
    
    df_fewshot = pd.DataFrame(results_fewshot)
    df_fewshot.to_excel(os.path.join(OUTPUT_DIR, "gpt4_d4_fewshot.xlsx"), index=False)

    # 5. Evaluation
    # Extract predictions ensuring types match
    zs_dep = df_zeroshot['Depression_Risk'].fillna(0).astype(int)
    zs_sui = df_zeroshot['Suicide_Risk'].fillna(0).astype(int)
    
    fs_dep = df_fewshot['Depression_Risk'].fillna(0).astype(int)
    fs_sui = df_fewshot['Suicide_Risk'].fillna(0).astype(int)

    print("\n=== Zero-shot Results ===")
    evaluate_performance(df_ground_truth['Depression_Risk'], zs_dep, "Depression Risk")
    evaluate_performance(df_ground_truth['Suicide_Risk'], zs_sui, "Suicide Risk")

    print("\n=== Few-shot Results ===")
    evaluate_performance(df_ground_truth['Depression_Risk'], fs_dep, "Depression Risk")
    evaluate_performance(df_ground_truth['Suicide_Risk'], fs_sui, "Suicide Risk")

if __name__ == "__main__":
    main()