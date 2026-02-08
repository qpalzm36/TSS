import json
import re
import pandas as pd
import os
from tqdm import tqdm


INPUT_FILE = ""
OUTPUT_DIR = ""
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test.parquet")

def parse_step_info(key):

    num_match = re.search(r"Step\s+(\d+)", key)
    step_num = int(num_match.group(1)) if num_match else -1
    tag = "<retrieval>" if "<retrieval>" in key else "<no-retrieval>"
    return step_num, tag

def split_content_and_example(value):

    if "<p>" in value:
        parts = value.split("<p>")
        reasoning = parts[0].strip()
        example = parts[1].split("</p>")[0].strip()
        return reasoning, example
    return value.strip(), None

def construct_tss_query(problem, history_steps):

    query = f"Problem: {problem}\n"
    for s in history_steps:
        query += f"{s}\n"
    return query.strip()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processed_data = []

    print(f"Reading SFT data from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for idx, line in tqdm(enumerate(lines), total=len(lines)):
        try:
            item = json.loads(line)
            problem = item['problem']
            
     
            step_keys = [k for k in item.keys() if 'Step' in k]
            step_keys.sort(key=lambda x: int(re.search(r'Step\s+(\d+)', x).group(1)))
            
            reference_trace = []
            history_reasoning = []
            final_answer = ""

            for key in step_keys:
                step_num, tag = parse_step_info(key)
                reasoning, golden_example = split_content_and_example(item[key])
                
      
                expected_query = None
                if tag == "<retrieval>":
                    expected_query = construct_tss_query(problem, history_reasoning)
                
               
                formatted_reasoning = json.dumps({f"Step {step_num}": reasoning}, ensure_ascii=False)

                reference_trace.append({
                    "step_idx": step_num,
                    "expected_tag": tag,
                    "expected_query": expected_query,
                    "expected_content": formatted_reasoning,
                    "golden_example": golden_example
                })
                
                history_reasoning.append(reasoning)
                
                if key == step_keys[-1]:
                    final_answer = reasoning.split("[CONCLUSION]")[-1].strip() if "[CONCLUSION]" in reasoning else reasoning

            system_msg = (
    "## Role\n"
    "You are an expert math solver using a step-by-step retrieval loop.\n\n"
    
    "## TSS Loop Logic\n"
    "Step 1 (Decision): Start with <no-retrieval> or <retrieval>.\n"
    "Step 2 (Search): If you choose <retrieval>, output <search>Problem + Last Step</search> and STOP. Wait for the system to provide a relevant example in <p>...</p> tags.\n"
    "Step 3 (Reasoning): After receiving the <p> example, use its logic to output the reasoning for the current step as a SINGLE JSON object.\n\n"
    
    "## One-Shot Example\n"
    "User: Find sqrt(64).\n"
    "Assistant (Search): <retrieval> <search>Find sqrt(64).</search>\n"
    "System Feedback: <p>{\"Problem\": \"Find sqrt(100)\", \"Step 1\": \"[CONDITION] 100=10^2. [PROCESS] sqrt(100)=10. [CONCLUSION] 10.\"}</p>\n"
    "Assistant (Reasoning): {\"Step 1\": \"[CONDITION] 64 = 8^2. [PROCESS] Use the property sqrt(x^2)=x. [CONCLUSION] sqrt(64) = 8. \\boxed{8} [END_OF_SOLUTION]\"}\n\n"
    
    "## Rules\n"
    "1. NO placeholders like 'Problem + Step'. Copy-paste the ACTUAL text into <search>.\n"
    "2. If you see a <p> block, you MUST use its logic to write the next JSON Step.\n"
    "3. Output ONLY ONE tag or ONE JSON per turn. STOP after '</search>' or '}'.\n"
    "4. Do NOT output 'User:', 'Assistant:', or 'System Feedback:' labels."
)
            
            rl_item = {
                "id": item.get('id', f"tss_{idx}"),
                "data_source": "math_tss",
                "ability": "math-reasoning",
                "prompt": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Problem: {problem}"}
                ],
                "reward_model": {
                    "style": "tss_single_stream",
                    "ground_truth": {
                        "final_answer": final_answer,
                        "reference_trace": reference_trace
                    }
                }
            }
            processed_data.append(rl_item)

        except Exception as e:
            print(f"Error processing line {idx}: {e}")

    df = pd.DataFrame(processed_data)
    df.to_parquet(OUTPUT_FILE)
    print(f"Successfully saved {len(df)} samples to {OUTPUT_FILE}")
    
    print("\n--- Processed Data Sample ---")
    print(json.dumps(processed_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()