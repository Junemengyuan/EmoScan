import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import classification_report

# ==========================================
# Configuration & Constants
# ==========================================

# API Configuration
API_KEY = "your_api_key_here"  # Recommend using os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# Paths - Update these before running
DATA_DIR = "./data/" 
OUTPUT_DIR = "./results/"
TRAIN_FILE = os.path.join(DATA_DIR, "train_ft_screening.json")
TEST_FILE = os.path.join(DATA_DIR, "test_ft_screening.json")
GROUND_TRUTH_FILE = os.path.join(OUTPUT_DIR, "grdtruth_augmented.xlsx")

# Mapping for fine-grained classification (0-indexed)
FINER_LABELS_MAP = {
    "major depressive disorder": 0,
    "persistent depressive disorder (dysthymia)": 1,
    "premenstrual dysphoric disorder": 2,
    "disruptive mood dysregulation disorder": 3,
    "depressive disorder due to another medical condition": 4,
    "substance/medication-induced depressive disorder": 5,
    "selective mutism": 6,
    "other specified depressive disorder": 7,
    "unspecified depressive disorder": 8,
    "social anxiety disorder": 9,
    "generalized anxiety disorder": 10,
    "separation anxiety disorder": 11,
    "panic disorder": 12,
    "agoraphobia": 13,
    "specific phobia": 14,
    "panic attack specifier": 15,
    "substance/medication-induced anxiety disorder": 16,
    "unspecified anxiety disorder": 17,
    "anxiety disorder due to another medical condition": 18,
    "other specified anxiety disorder": 19,
}

# ==========================================
# Prompts
# ==========================================

INSTRUCTION_DSM5 = """Your task is to screen for emotional disorders based on DSM-5, and provide a discussion. 
    
Here, the emotional disorders consist of Anxiety Disorders and Depressive Disorders. 
Under Anxiety Disorders, there are social anxiety disorder, generalized anxiety disorder, separation anxiety disorder, panic disorder, agoraphobia, specific phobia, 
selective mutism, panic attack specifier, substance/medication-induced anxiety disorder, unspecified anxiety disorder, anxiety disorder due to another medical condition,
and other specified anxiety disorder. 
Under Depressive Disorders, there are major depressive disorder, persistent depressive disorder (dysthymia), premenstrual dysphoric disorder, 
disruptive mood dysregulation disorder, depressive disorder due to another medical condition, substance/medication-induced depressive disorder, 
other specified depressive disorder, and unspecified depressive disorder.
"""

COT_PROMPT = """
To consider if the client has an emotional disorder, and the category of the emotional disorder, we should first follow:
If the client is depressed or experiences the client experiences loss of pleasure, we should consider A. possibilities of depressive disorders (but does not mean the client has depressive disorders. You still have some steps to follow). 
If the client has fear anxiety, panic, or continuing worry, we should consider B. possibilities of anxiety disorders (but does not mean the client has anxiety disorders. You still have some steps to follow). 

A. Depressive Disorders:
1. Check for history of mania, hypomania, or mixed state (exclude depressive and anxiety disorders).
2. Assess for disruptive mood dysregulation disorder if chronic, severe persistent irritability is present.
3. Consider depressive disorder due to another medical condition if a significant medical condition is present.
4. Evaluate for substance/medication-induced depressive disorder if significant recent substance use is reported.
5. Rule out emotional disorders if a history of many somatic symptoms exists.
6. Check for premenstrual dysphoric disorder if premenstrual symptoms are reported.
7. Consider major depressive disorder if symptoms meet the criteria, without hallucinations or delusions.
8. Assess for persistent depressive disorder (dysthymia) if symptoms of dysthymia are present.
9. Consider other specified depressive disorder if depressive symptoms don't meet full criteria.
10. Evaluate for unspecified depressive disorder if mood causes distress or impairment.
11. If none of the above apply, it's likely the client has no emotional disorders.

B. Anxiety Disorders:
1. Consider anxiety disorder due to another medical condition if a significant medical condition is present.
2. Evaluate for substance/medication-induced anxiety disorder if significant recent substance use is reported.
3. Rule out emotional disorders if anxiety is related to a highly stressful event, substance use, compulsion/obsession, or worry.
4. Assess for generalized anxiety disorder if chronic worry and associated symptoms are present.
5. Consider social anxiety disorder if fear is related to embarrassment in performance/social situations.
6. Assess for selective mutism in children who consistently fail to speak in specific social situations.
7. Evaluate for specific phobia if fear is specific to certain situations or objects.
8. Consider separation anxiety disorder if there is inappropriate and excessive fear or anxiety concerning separation.
9. Assess for agoraphobia and panic disorder if fear is related to being away from home or places where escape is difficult, with recurring surprise panic attacks.
10. Assess for agoraphobia if fear is related to being away from home or places where escape is difficult, without recurring surprise panic attacks.
11. If none of the above apply, evaluate for panic disorder if there are recurring surprise panic attacks.
12. Rule out emotional disorders if anxiety is caused by a non-extreme stressful event without reexperiencing, arousal, or avoidance.
13. Consider other specified anxiety disorder if anxiety causes distress or impairment without meeting full criteria.
14. If none of the above apply, it's likely the client has no emotional disorders.

In our sample, there are about 10%-15% of people who have emotional disorders.

Here is a conversation between a psychiatrist and a client. You should read the conversation and provide an output according to the format requirement.

<start_of_conversation>
Client: I'm at the end of my rope with stress; freaking out over my master's finals while juggling work and being a single mom. It feels like too much.
Psychiatrist: It sounds like you're carrying a heavy load. Could you share if you've felt this overwhelmed with stress or anxiety before, or is this a new intensity?
Client: This is definitely the first time it's hit me this hard. I used to manage somehow, but ever since my husband passed and the student loans kicked in, I'm swamped.
Psychiatrist: I'm deeply sorry for your loss. That's a considerable burden to bear on top of everything else. Have you ever sought help for stress like this before, or struggled with drugs or alcohol?
Client: No, I've never had to seek help for stress or anything before, and I steer clear of drugs and alcohol. This level of stress is new territory for me.
Psychiatrist: It's beneficial that you don't have those additional challenges to handle. Regarding your health, are there any conditions or concerns we should know about?
Client: No, thankfully, I don't have any health problems ongoing.
Psychiatrist: I appreciate you sharing that. Shifting focus slightly, I remember you mentioned your family lives a fair distance away. Is there a history of emotional disorders in your family?
Client: No, my family doesn’t have any history of that sort; we're pretty standard, just live far apart.
Psychiatrist: Understanding your family context helps frame things. Diving into your personal context a bit more — you've got a lot on your plate with work, parenting alone, and studying. Could you expand on your support network, and any major stresses or traumas, like your husband's loss?
Client: My social circle's pretty small since my family's far and I'm all about my daughters, work, and school. Losing my husband in a car accident was a watershed moment for us — it's been really tough.
Psychiatrist: Handling all of that without much support is undeniably daunting. Your willingness to discuss these challenges today is commendable. For now, we'll conclude here, but remember, reaching out for help signifies strength. We'll review everything you've shared and outline the next steps. Thank you for being so open.
</end_of_conversation>

First, you should screen if the client has any emotional disorder based on DSM-5 according to the conversation.

If the client has emotional disorder, your output format should be:
Emotional disorder screening: Yes
Screening result (coarse): xxx
Screening result (fine): xxx
Discussion (fine): xxx

If the client does not have any emotional disorder, your output format should be:
Emotional disorder screening: No
Discussion: xxx

Let’s think step by step, with rules listed above:
1. The client discusses feeling overwhelmed with stress and anxiety related to their master's finals, work, and being a single mom.
2. Rule B from the Anxiety Disorders section applies, as the client mentions experiencing anxiety.
3. None of the specific anxiety disorders listed in rules 4-10 apply to the client's situation.
4. Rule 11 states that if none of the above apply, evaluate for panic disorder if there are recurring surprise panic attacks. However, the client does not mention experiencing panic attacks.
5. Rule 13 suggests considering other specified anxiety disorder if anxiety causes distress or impairment without meeting full criteria. However, the client's symptoms do not meet the criteria for any specific anxiety disorder.
6. Rule 14 states that if none of the above apply, it's likely the client has no emotional disorders.
7. The screening result is "No" for emotional disorder screening.

Emotional disorder screening: No
Discussion: Upon reviewing the conversation, the client does not satisfy the diagnostic criteria for any emotional disorder as delineated in the Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition (DSM-5) pertaining to emotional disorders. The symptoms presented by the client do not reach the diagnostic threshold required for a clinical diagnosis. Should the client encounter any emotional challenges, seeking further support from certified peer supporters or licensed professional counselors is recommended.
"""

RULES_COT_FEWSHOT = """To consider if the client has an emotional disorder, and the category of the emotional disorder, we should first follow:
If the client is depressed or experiences the client experiences loss of pleasure, we should consider A. possibilities of depressive disorders (but does not mean the client has depressive disorders. You still have some steps to follow). 
If the client has fear anxiety, panic, or continuing worry, we should consider B. possibilities of anxiety disorders (but does not mean the client has anxiety disorders. You still have some steps to follow). 
[... Rules A and B as defined above ...]
In our sample, there are about 10%-15% of people who have emotional disorders.
"""

STEPS_LIST = [
    # Example 1 Reasoning
    """Let's think step by step, with rules listed above:
1. The client discusses feeling overwhelmed with stress and anxiety related to their master's finals, work, and being a single mom.
2. Rule B from the Anxiety Disorders section applies, as the client mentions experiencing anxiety.
3. None of the specific anxiety disorders listed in rules 4-10 apply to the client's situation.
4. Rule 11 states that if none of the above apply, evaluate for panic disorder if there are recurring surprise panic attacks. However, the client does not mention experiencing panic attacks.
5. Rule 13 suggests considering other specified anxiety disorder if anxiety causes distress or impairment without meeting full criteria. However, the client's symptoms do not meet the criteria for any specific anxiety disorder.
6. Rule 14 states that if none of the above apply, it's likely the client has no emotional disorders.
7. The screening result is "No" for emotional disorder screening.
    """,
    # Example 2 Reasoning
    """Let’s think step by step, with rules listed above:
1. The client experiences fear and anxiety, as mentioned in the conversation.
2. Rule B from the Anxiety Disorders section applies, as fear and anxiety are present.
3. Consider social anxiety disorder if fear is related to embarrassment in performance/social situations. This rule is relevant because the client mentions being nervous around people, fear of criticism, and avoiding the spotlight, which are characteristics of social anxiety disorder.
4. Based on the information provided, the client's symptoms align with social anxiety disorder, which involves fear of social scrutiny, fear of negative evaluation, and persistent anxiety in social situations.
5. The screening result (coarse) is Anxiety Disorders.
6. The screening result (fine) is social anxiety disorder.
    """,
    # Example 3 Reasoning
    """Let’s think step by step, with rules listed above:
1. The client mentions feeling sad and alone, indicating the possibility of a depressive disorder.
2. Rule A from the Depressive Disorders section applies, as the client mentions feeling depressed.
3. None of the specific depressive disorders listed in rules 1-10 apply to the client's situation.
4. Rule 11 states that if none of the above apply, it's likely the client has no emotional disorders. However, the client's symptoms of depression are persistent and severe.
5. Rule B from the Anxiety Disorders section doesn't apply as the client does not mention fear, anxiety, panic, or continuing worry.
6. The screening result is "Yes" for emotional disorder screening, indicating the presence of an emotional disorder.
7. The screening result (coarse) is "Depressive Disorders."
8. The screening result (fine) is "Major Depressive Disorder," as the client reports persistently depressed mood, insomnia, fatigue, feelings of worthlessness, and suicidality.
""",
    # Example 4 Reasoning
    """Let’s think step by step, with rules listed above:
1. The client mentions feeling like quitting college, having a hard time with classes, and experiencing sleep difficulties and difficulty focusing. These symptoms indicate the possibility of an emotional disorder.
2. Rule A from the Depressive Disorders section applies, as the client mentions feeling depressed and experiencing a loss of pleasure in their activities.
3. None of the specific depressive disorders listed in rules 1-10 apply to the client's situation.
4. Rule 11 states that if none of the above apply, it's likely the client has no emotional disorders.
5. The screening result is "No" for emotional disorder screening, indicating that the client does not satisfy the diagnostic criteria for any emotional disorder as delineated in the DSM-5 pertaining to emotional disorders.
"""
]

# ==========================================
# Inference Functions
# ==========================================

def get_llm_response(system_role, prompt, model="gpt-4"):
    """Calls OpenAI API."""
    try:
        messages = []
        if system_role:
            messages.append({"role": "system", "content": system_role})
        messages.append({"role": "user", "content": prompt})
        
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return ""

def build_fewshot_prompt(train_data, indices=[0, 5, 507, 527], use_cot=False):
    """Constructs few-shot examples from training data."""
    examples = ""
    if use_cot:
        examples += RULES_COT_FEWSHOT
        
    prompt_intro = "Here are some examples you can refer to: \n\n"
    examples += prompt_intro
    
    for i, idx in enumerate(indices):
        examples += f"<start_of_Example_{i+1}>:\n"
        examples += train_data[idx]['messages'][0]['content']
        if use_cot:
            examples += "\n" + STEPS_LIST[i]
        examples += "\n" + train_data[idx]['messages'][1]['content']
        examples += f"\n<end_of_Example_{i+1}>\n\n"
        
    examples += "Now please do the following task: \n\n"
    return examples

def parse_response(response_text):
    """Parses the structured output from LLM."""
    if "step by step" in response_text:
        # Extract content after reasoning
        try:
            response_text = 'Emotional disorder screening' + response_text.split('Emotional disorder screening')[1]
        except IndexError:
            pass # Handle malformed COT output

    lines = response_text.split('\n')
    clean_lines = [line.split(": ")[1] if ": " in line else line for line in lines]
    
    # Basic structure reconstruction
    result = {
        "emotional_disorder_screening": 0,
        "screening_coarse": "NA",
        "screening_fine": "NA",
        "discussion": ""
    }

    if clean_lines:
        if "Yes" in clean_lines[0]:
            result["emotional_disorder_screening"] = 1
            if len(clean_lines) > 1: result["screening_coarse"] = clean_lines[1]
            if len(clean_lines) > 2: result["screening_fine"] = clean_lines[2]
            if len(clean_lines) > 3: result["discussion"] = "".join(clean_lines[3:])
        elif "No" in clean_lines[0]:
            result["emotional_disorder_screening"] = 0
            if len(clean_lines) > 1: result["discussion"] = "".join(clean_lines[1:])
            
    return result

def run_inference(test_data, system_prompt, fewshot_prefix=""):
    """Main inference loop."""
    results = []
    
    # Using a subset for demonstration/testing if needed
    for item in tqdm(test_data):
        query = item['messages'][0]['content']
        full_prompt = fewshot_prefix + query
        
        # If using CoT (empty system prompt usually passed in original code for CoT)
        if system_prompt == "":
            response = get_llm_response("", full_prompt)
        else:
            response = get_llm_response(system_prompt, full_prompt)
            
        parsed = parse_response(response)
        results.append(parsed)
        
    return pd.DataFrame(results)

# ==========================================
# Evaluation Metrics
# ==========================================

def get_coarse_labels(df, col_name):
    """Converts text labels to coarse numerical vectors [Depression, Anxiety]."""
    output = []
    for val in df[col_name]:
        val_str = str(val)
        res = [0, 0]
        if "epress" in val_str:
            res[0] = 1
        if "nxiet" in val_str:
            res[1] = 1
        output.append(res)
    return output

def get_fine_labels(df, col_name):
    """Converts text labels to fine-grained multilabel vectors."""
    num_classes = len(FINER_LABELS_MAP)
    output = []
    
    for val in df[col_name]:
        multilabel = [0] * num_classes
        if isinstance(val, str):
            val_lower = val.lower()
            for label, idx in FINER_LABELS_MAP.items():
                if label in val_lower:
                    multilabel[idx] = 1
        output.append(multilabel)
    return output

def print_evaluation(ground_truth_df, pred_df, title="Evaluation"):
    print(f"--- {title} ---")
    
    # Coarse Evaluation
    gt_coarse = get_coarse_labels(ground_truth_df, "screening_coarse")
    pred_coarse = get_coarse_labels(pred_df, "screening_coarse")
    
    # Filter for positive cases (if coarse is not [0,0]) for detailed metrics
    pos_indices = [i for i, x in enumerate(gt_coarse) if x != [0, 0]]
    if pos_indices:
        gt_pos = [gt_coarse[i] for i in pos_indices]
        pred_pos = [pred_coarse[i] for i in pos_indices]
        print("\nCoarse Classification (Positive Cases Only):")
        print(classification_report(gt_pos, pred_pos, zero_division=0))
    
    print("\nCoarse Classification (All):")
    print(classification_report(gt_coarse, pred_coarse, zero_division=0))

    # Fine-grained Evaluation
    gt_fine = get_fine_labels(ground_truth_df, "screening_fine")
    pred_fine = get_fine_labels(pred_df, "screening_fine")
    
    print("\nFine-grained Classification:")
    print(classification_report(gt_fine, pred_fine, zero_division=0))

# ==========================================
# Main Execution
# ==========================================

def main():
    # 1. Load Data
    if not os.path.exists(TEST_FILE):
        print(f"File not found: {TEST_FILE}")
        return

    with open(TEST_FILE, 'r') as f:
        test_data = json.load(f)
    
    with open(TRAIN_FILE, 'r') as f:
        train_data = json.load(f)

    # Load or Create Ground Truth
    if os.path.exists(GROUND_TRUTH_FILE):
        df_ground_truth = pd.read_excel(GROUND_TRUTH_FILE)
    else:
        print("Generating ground truth from test file...")
        # Logic to extract ground truth from test_ft_screening.json structure
        gt_results = []
        for item in test_data:
            raw_answer = item['messages'][1]['content']
            parsed = parse_response(raw_answer)
            gt_results.append(parsed)
        df_ground_truth = pd.DataFrame(gt_results)
        df_ground_truth.to_excel(GROUND_TRUTH_FILE, index=False)

    # 2. Run Experiments
    
    # A. Zero-shot
    print("Running Zero-shot...")
    df_zeroshot = run_inference(test_data, INSTRUCTION_DSM5)
    df_zeroshot.to_excel(os.path.join(OUTPUT_DIR, "gpt4_zeroshot.xlsx"), index=False)
    print_evaluation(df_ground_truth, df_zeroshot, "Zero-Shot GPT-4")

    # B. Few-shot
    print("Running Few-shot...")
    fewshot_prompt = build_fewshot_prompt(train_data, use_cot=False)
    df_fewshot = run_inference(test_data, INSTRUCTION_DSM5 + fewshot_prompt)
    df_fewshot.to_excel(os.path.join(OUTPUT_DIR, "gpt4_fewshot.xlsx"), index=False)
    print_evaluation(df_ground_truth, df_fewshot, "Few-Shot GPT-4")

    # C. Zero-shot Chain-of-Thought
    print("Running CoT...")
    df_cot = run_inference(test_data, "", fewshot_prefix=COT_PROMPT)
    df_cot.to_excel(os.path.join(OUTPUT_DIR, "gpt4_cot.xlsx"), index=False)
    print_evaluation(df_ground_truth, df_cot, "CoT GPT-4")

    # D. Few-shot Chain-of-Thought
    print("Running Few-shot CoT...")
    fewshot_cot_prompt = build_fewshot_prompt(train_data, use_cot=True)
    df_fewshot_cot = run_inference(test_data, INSTRUCTION_DSM5 + fewshot_cot_prompt)
    df_fewshot_cot.to_excel(os.path.join(OUTPUT_DIR, "gpt4_fewshot_cot.xlsx"), index=False)
    print_evaluation(df_ground_truth, df_fewshot_cot, "Few-Shot CoT GPT-4")

if __name__ == "__main__":
    main()