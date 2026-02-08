import json
import os
import torch
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import argparse

# --- Configuration ---
# 请在这里填入您的OpenAI API密钥
# IMPORTANT: Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = ""


openai.api_key = os.environ["OPENAI_API_KEY"]
try:
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.chatanywhere.tech/v1")
except TypeError:
    print("错误:OPENAI_API_KEY 未设置。请在您的环境中设置该变量。")
    exit(1)

# 默认路径
MODEL_PATH = ""
FAISS_INDEX_PATH = ""
KNOWLEDGE_BASE_DOCS_PATH = ""
INPUT_DATA_PATH = ""
OUTPUT_DATA_PATH = ""
OUTPUT_CHART_PATH = ""

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


# --- Helper Functions ---

def get_consistency_score(gold_step_content: str, retrieved_step_content: str) -> int:
    """
    Calls GPT-4o-mini to evaluate the logical consistency between the gold and retrieved steps.
    """
    prompt = f"""
    You are an expert in mathematical reasoning. Your task is to evaluate if a "Retrieved Step" provides a logically sound example for deriving a "Gold Step".

    Both steps are self-contained with [CONDITION], [PROCESS], and [CONCLUSION]. Your evaluation must focus on the similarity of the logical leap in the [PROCESS], given the context in [CONDITION].

    **Part 1: Scoring Rubric**

    - **5 (Perfect Analogy):** The Retrieved Step solves a problem of the *exact same type* using the *exact same method*. A student can directly map their problem onto the example by just changing numbers.
    - **4 (Method Demonstration):** The Retrieved Step demonstrates the *core method or formula* required, but the *problem setup or context is different*. The student needs an insight to see how the method applies to their specific problem.
    - **3 (Fair):** The Retrieved Step demonstrates a partially related concept or a sub-step, but not the main logical leap.
    - **2 (Poor):** The Retrieved Step is on the same general topic but uses an inapplicable or incorrect method for the goal. It's more likely to confuse than help.
    - **1 (Irrelevant):** The logic and mathematical domain are completely unrelated.

    **Part 2: Examples for Calibration**

    --- Example for Score 5 (Perfect Analogy) ---
    Gold Step: "[CONDITION] There are 10 red balls and 10 blue balls in a bag. We draw two balls without replacement. [PROCESS] The probability of the first being red is 10/20. The probability of the second being red is then 9/19. The total probability is (10/20) * (9/19). [CONCLUSION] The probability is 9/38."
    Retrieved Step: "[CONDITION] A batch has 50 items, 5 of which are defective. We pick two items without replacement. [PROCESS] The probability of the first being defective is 5/50. The probability of the second being defective is 4/49. The total probability is (5/50) * (4/49). [CONCLUSION] The probability is 2/245."
    Reasoning: Both problems are of the exact same type (conditional probability without replacement). Perfect 5.

    --- Example for Score 4 (Method Demonstration) ---
    Gold Step: "[CONDITION] The radius of a circle is 5 and the distance from the center to a chord is 3. This forms a right triangle with the radius as the hypotenuse. [PROCESS] Use the Pythagorean theorem to find half the chord: semi-chord = sqrt(5^2 - 3^2). [CONCLUSION] The semi-chord is 4."
    Retrieved Step: "[CONDITION] A standard right triangle has a hypotenuse of 13 and one leg of 5. [PROCESS] To find the other leg, use the Pythagorean theorem: leg = sqrt(13^2 - 5^2). [CONCLUSION] The other leg is 12."
    Reasoning: Demonstrates the necessary *method* (Pythagorean theorem) but not in the same *problem context* (circle geometry vs. basic triangle). Strong 4.

    --- Example for Score 3 (Fair) ---
    Gold Step: "[CONDITION] We need to solve the inequality x^2 - 4x + 3 < 0. [PROCESS] First, find the roots of x^2 - 4x + 3 = 0, which are x=1 and x=3. Then test the intervals. [CONCLUSION] The solution is 1 < x < 3."
    Retrieved Step: "[CONDITION] We need to solve the equation x^2 - 5x + 6 = 0. [PROCESS] Factor it as (x-2)(x-3) = 0. [CONCLUSION] The roots are x=2 and x=3."
    Reasoning: Demonstrates a necessary sub-step (finding roots) but fails to address the main goal (solving the inequality). Fair 3.

    --- Example for Score 2 (Poor) ---
    Gold Step: "[CONDITION] We need the volume of a cylinder with radius r=3 and height h=10. [PROCESS] The volume formula is V = pi * r^2 * h = pi * 3^2 * 10. [CONCLUSION] The volume is 90pi."
    Retrieved Step: "[CONDITION] We need a property of a cylinder with radius r=3 and height h=10. [PROCESS] The surface area formula is A = 2*pi*r*h + 2*pi*r^2 = 60pi + 18pi. [CONCLUSION] The surface area is 78pi."
    Reasoning: The topic (cylinder) is the same, but the retrieved step uses the wrong formula (surface area instead of volume). This is confusing and unhelpful. Clear 2.

    --- Example for Score 1 (Irrelevant) ---
    Gold Step: "[CONDITION] We need the probability of drawing two aces from a 52-card deck. [PROCESS] P(A and B) = (4/52) * (3/51). [CONCLUSION] The probability is 1/221."
    Retrieved Step: "[CONDITION] The function is f(x) = x^3. [PROCESS] Apply the power rule for derivatives. [CONCLUSION] The derivative is f'(x) = 3x^2."
    Reasoning: Completely unrelated mathematical domains (probability vs. calculus). Obvious 1.

    **Part 3: Your Task**

    Analyze the two steps below. Your response MUST be a single integer from 1 to 5.

    ---
    **Gold Step:**
    {gold_step_content}
    ---
    **Retrieved Step:**
    {retrieved_step_content}
    ---

    **Your Score (1-5):**
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert mathematical reasoning judge. Evaluate the logical similarity of two provided solution steps on a scale of 1 to 5 based on the provided rubric and examples."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5,
        )
        score_text = response.choices[0].message.content.strip()
        # Extract the first integer found in the response
        match = re.search(r'\d+', score_text)
        if match:
            return int(match.group(0))
        else:
            print(f"Warning: Could not parse score from response: '{score_text}'. Defaulting to 1.")
            return 1
    except Exception as e:
        print(f"An error occurred during OpenAI API call: {e}")
        return 1 # Return a default low score on error

def find_previous_step_content(data: dict, current_step_num: int) -> str:
    """Finds the content of the step immediately preceding the current one."""
    prev_step_num = current_step_num - 1
    
    # Check for both <retrieval> and <no-retrieval> tags
    possible_keys = [
        f"<retrieval>Step {prev_step_num}",
        f"<no-retrieval>Step {prev_step_num}"
    ]
    
    for key in possible_keys:
        if key in data:
            return data[key]
            
    # Fallback for keys that might not have tags (e.g., "Step 1")
    if f"Step {prev_step_num}" in data:
        return data[f"Step {prev_step_num}"]
        
    return ""

# --- Main Script ---

def main():
    parser = argparse.ArgumentParser(description="Build generator data with retrieval and quality filtering.")
    # ... (parser arguments remain the same) ...
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the finetuned retriever model")
    parser.add_argument("--faiss_index_path", type=str, default=FAISS_INDEX_PATH, help="Path to the FAISS index file")
    parser.add_argument("--knowledge_base_docs_path", type=str, default=KNOWLEDGE_BASE_DOCS_PATH, help="Path to the knowledge base documents file")
    parser.add_argument("--input_data_path", type=str, default=INPUT_DATA_PATH, help="Path to the input data file")
    parser.add_argument("--output_data_path", type=str, default=OUTPUT_DATA_PATH, help="Path to save the processed data file")
    parser.add_argument("--output_chart_path", type=str, default=OUTPUT_CHART_PATH, help="Path to save the performance chart")
    args = parser.parse_args()


    model_path = args.model_path
    faiss_index_path = args.faiss_index_path
    knowledge_base_docs_path = args.knowledge_base_docs_path
    input_data_path = args.input_data_path
    output_data_path = args.output_data_path
    output_chart_path = args.output_chart_path

    print("Step 1: Loading resources...")
    retriever_model = SentenceTransformer(model_path, device=DEVICE)
    faiss_index = faiss.read_index(faiss_index_path)
    
    with open(knowledge_base_docs_path, 'r', encoding='utf-8') as f:
        knowledge_base_docs = [line.strip() for line in f]
    
    print("Step 2: Processing data, performing retrieval, and filtering by quality score...")
    consistency_scores = []
    processed_data = []
    
    retrieval_failure_count = 0  # Structural failures
    semantic_failure_count = 0 # New: Failures due to low score
    total_retrieval_attempts = 0

    with open(input_data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing Problems"):
            data = json.loads(line)
            new_data_item = data.copy()

            sorted_keys = sorted(data.keys(), key=lambda k: int(re.search(r'Step (\d+)', k).group(1)) if 'Step' in k else -1)

            for key in sorted_keys:
                if key.startswith("<retrieval>Step"):
                    total_retrieval_attempts += 1
                    match = re.search(r'Step (\d+)', key)
                    step_num = int(match.group(1))
                    
                    problem_text = data['problem']
                    query_obj = {"Problem": problem_text}
                    if step_num > 1:
                        prev_step_content = find_previous_step_content(data, step_num)
                        if prev_step_content:
                            query_obj[f"Step {step_num - 1}"] = prev_step_content
                    query_str = json.dumps(query_obj)
                    
                    query_embedding = retriever_model.encode(query_str, convert_to_tensor=True, device=DEVICE)
                    query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
                    D, I = faiss_index.search(query_embedding_np, k=2)

                    valid_retrieved_doc_json_str = None
                    expected_num_keys = len(query_obj.keys()) + 1
                    for retrieved_idx in I[0]:
                        if retrieved_idx == -1: continue
                        try:
                            doc_json_str = knowledge_base_docs[retrieved_idx]
                            retrieved_doc = json.loads(doc_json_str)
                            if len(retrieved_doc.keys()) >= expected_num_keys:
                                valid_retrieved_doc_json_str = doc_json_str
                                break
                        except (json.JSONDecodeError, IndexError) as e:
                            print(f"\nWarning: Could not process retrieved document at index {retrieved_idx}. Error: {e}")
                            continue
                    
                    if valid_retrieved_doc_json_str:
                        gold_step_content = data[key]
                        retrieved_doc = json.loads(valid_retrieved_doc_json_str)
                        retrieved_step_content = list(retrieved_doc.values())[-1]
                        
                        score = get_consistency_score(gold_step_content, retrieved_step_content)
                        consistency_scores.append(score) # Always record score for stats

                        # --- START OF MODIFICATION: Quality Filtering ---
                        if score >= 3:
                            # Score is good, inject the retrieval context
                            retrieved_paragraph = f"<p>{valid_retrieved_doc_json_str}</p>"
                            new_data_item[key] += retrieved_paragraph
                        else:
                            # Score is poor, do not inject and count as a semantic failure
                            semantic_failure_count += 1
                        # --- END OF MODIFICATION ---
                    else:
                        retrieval_failure_count += 1
            
            last_step_key = None
            for key in reversed(sorted_keys):
                if 'Step' in key:
                    last_step_key = key
                    break
            if last_step_key:
                new_data_item[last_step_key] += " [END_OF_SOLUTION]"
            processed_data.append(new_data_item)

    print(f"\nStep 3: Saving processed data to {output_data_path}...")
    with open(output_data_path, 'w', encoding='utf-8') as f_out:
        for item in processed_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("\nStep 4: Analyzing and visualizing retrieval performance...")
    print(f"\n--- Retrieval Funnel Report ---")
    if total_retrieval_attempts > 0:
        structurally_valid_count = total_retrieval_attempts - retrieval_failure_count
        semantically_valid_count = structurally_valid_count - semantic_failure_count
        
        print(f"Total Retrieval Attempts: {total_retrieval_attempts}")
        print(f"  - Structurally Invalid (failed validation): {retrieval_failure_count}")
        print(f"  = Structurally Valid: {structurally_valid_count} ({(structurally_valid_count/total_retrieval_attempts)*100:.2f}%)")
        print(f"    - Semantically Invalid (Score < 3, filtered out): {semantic_failure_count}")
        print(f"    = Final Usable Retrievals (Injected into data): {semantically_valid_count} ({(semantically_valid_count/structurally_valid_count)*100 if structurally_valid_count>0 else 0:.2f}% of valid)")
    else:
        print("No retrieval attempts were made.")

    if not consistency_scores:
        print("\nNo structurally valid retrievals were made. No consistency scores to report.")
        return

    average_score = np.mean(consistency_scores)
    total_scored_retrievals = len(consistency_scores)
    relevant_retrievals_count = sum(1 for score in consistency_scores if score >= 3)
    relevance_percentage = (relevant_retrievals_count / total_scored_retrievals) * 100 if total_scored_retrievals > 0 else 0

    print(f"\n--- Logical Consistency Report (on all structurally valid retrievals) ---")
    print(f"Total Scored Retrievals: {total_scored_retrievals}")
    print(f"Average Logical Consistency Score: {average_score:.2f} / 5.0")
    print(f"Retrievals Considered 'Relevant' (Score >= 3): {relevant_retrievals_count} / {total_scored_retrievals} ({relevance_percentage:.2f}%)")
    
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    score_counts = {i: consistency_scores.count(i) for i in range(1, 6)}
    plt.bar(score_counts.keys(), score_counts.values(), color='skyblue', edgecolor='black')
    
    plt.title('Distribution of Retrieval Logical Consistency Scores')
    plt.xlabel('Consistency Score (1-5)')
    plt.ylabel('Number of Retrievals')
    plt.xticks(range(1, 6))
    plt.grid(axis='y', linestyle='--')

    for score, count in score_counts.items():
        if count > 0:
            plt.text(score, count + 0.1, str(count), ha='center', va='bottom')

    plt.savefig(output_chart_path)
    print(f"Performance chart saved to {output_chart_path}")

if __name__ == "__main__":
    main()
