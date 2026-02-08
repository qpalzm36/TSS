import json
import pandas as pd
import os

def make_test_rl_data(test_jsonl_path, output_parquet_path):

    system_msg = (
    "## Background\n"
    "You are an expert math solver using a 'step-wise decision-retrieval-reasoning' loop logic. "
    "Your goal is to solve math problems step-by-step with extreme precision.\n\n"
    
    "## Task Instruction (Planner Mode)\n"
    "For EACH step, you MUST start your response with a decision tag:\n"
    "1. If you can solve the current step by yourself, output: <no-retrieval>\n"
    "2. If you need logic guidance, output: <retrieval> <search>Problem + Last Step Content</search>\n\n"
    
    "### How to construct your Search Query (TSS Markovian Logic):\n"
    "- If you are at Step 1: Your <search> query MUST be the full [Problem] statement.\n"
    "- If you are at Step N (N > 1): Your <search> query MUST be the [Problem] statement concatenated with the full content of the [Step N-1] you just completed.\n\n"
    
    "### Example Pattern:\n"
    "<retrieval> <search>[Problem content] + [Step N-1 content]</search> {\"Step N\": \"[CONDITION]... [PROCESS]... [CONCLUSION]...\"}\n\n"
    
    "## Format Requirement\n"
    "1. CRITICAL: Output the tag (<no-retrieval> or <retrieval> <search>...</search>) FIRST, then the {JSON} object.\n"
    "2. JSON key must follow the sequence: \"Step 1\", \"Step 2\", etc.\n"
    "3. Final answer: Include \\boxed{answer} inside the [CONCLUSION] section of the final step.\n"
    "4. Add the token [END_OF_SOLUTION] immediately after the final JSON object.\n"
    "5. Output ONLY the tags and the JSON. Do NOT write 'user' or 'assistant' labels."
)

    test_data = []
    with open(test_jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            test_data.append({
                "id": item.get('unique_id', f"test_{idx}"),
                "data_source": "math_standard_test", 
                "ability": "math-reasoning",
                "prompt": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Problem: {item['problem']}"}
                ],
                "reward_model": {
                    "style": "rule", 
                    "ground_truth": {
                        "target": item['answer'] 
                    }
                }
            })
    
    pd.DataFrame(test_data).to_parquet(output_parquet_path)

if __name__ == "__main__":

    test_jsonl = ""
    make_test_rl_data(test_jsonl, "")
    
    print("")