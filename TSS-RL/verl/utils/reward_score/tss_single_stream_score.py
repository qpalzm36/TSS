import re
import json
import numpy as np
from verl.utils.reward_score.qa_em import normalize_answer, em_check

def remove_latex_formatting(text):

    if not text: return ""
    text = str(text)

    text = text.replace('$', '')

    text = text.replace('\\text{', '').replace('}', '')

    text = text.replace('[END_OF_SOLUTION]', '').strip()
    return text

def extract_model_answer(solution_str):


    matches = re.findall(r'\\boxed\{(.*?)\}', solution_str)
    if matches:
        return matches[-1]
    
    
    if "[CONCLUSION]" in solution_str:
        try:
            part = solution_str.split("[CONCLUSION]")[-1]
            part = part.split("}")[0] 
            return part.replace("[END_OF_SOLUTION]", "").strip()
        except: pass
    return ""

def clean_ground_truth_answer(gt_str):

    if not isinstance(gt_str, str): return str(gt_str)
    

    matches = re.findall(r'\\boxed\{(.*?)\}', gt_str)
    if matches:
        return matches[-1]
    

    cleaned = gt_str.replace('$', '').replace('[END_OF_SOLUTION]', '').strip()
    

    if "=" in cleaned:
        cleaned = cleaned.split("=")[-1].strip()
        
    return cleaned
def calculate_similarity(a, b):
    if not a or not b: return 0.0
    a_set = set(normalize_answer(str(a)).split())
    b_set = set(normalize_answer(str(b)).split())
    if not a_set: return 0.0
    overlap = len(a_set.intersection(b_set))
    return overlap / max(len(a_set), len(b_set))

def is_valid_step_format(step_str):
    try:
        data = json.loads(step_str)
        content = list(data.values())[0]
        if not isinstance(content, str):
            return False, "Not string", str(content)
    except:
        return False, "Invalid JSON", ""
    required_tags = ["[CONDITION]", "[PROCESS]", "[CONCLUSION]"]
    if not all(tag in content for tag in required_tags):
        return False, "Missing CPC tags", content
    return True, "Valid", content

def compute_score(solution_str, ground_truth, method='strict', 
                  format_score=0.1, score=1.0,
                  structure_format_score=0.2, 
                  final_format_score=1.0, 
                  retrieval_score=0.3):

    total_reward = 0.0
    sub_scores = {
        'reward/exact_match': 0.0,          
        'reward/format_correctness': 0.0,   
        'reward/decision_accuracy': 0.0,    
        'reward/query_concatenation': 0.0,  
        'reward/content_similarity': 0.0,   
        'reward/retrieval_utilization': 0.0, 
        'reward/total_reward': 0.0
    }


    if isinstance(ground_truth, str):
        try: ground_truth = json.loads(ground_truth.replace("'", '"'))
        except: return 0.0, sub_scores

    ref_trace = ground_truth.get('reference_trace', [])
    final_answer_gt = ground_truth.get('final_answer', "")
    problem_text = ground_truth.get('problem', "")

    if len(ref_trace) == 0: return 0.0, sub_scores


    has_tag = "<retrieval>" in solution_str or "<no-retrieval>" in solution_str
    sub_scores['reward/format_correctness'] = 0.2 if has_tag else -0.2
    
    if "[END_OF_SOLUTION]" in solution_str:
        sub_scores['reward/format_correctness'] += 0.3
    else:
        total_reward -= 0.1 


    raw_steps = re.findall(r'\{.*?\}', solution_str, re.DOTALL)
    num_trace = len(ref_trace)

    for i, ref in enumerate(ref_trace):
        if i >= len(raw_steps): break
        is_valid, _, clean_content = is_valid_step_format(raw_steps[i])
        if not is_valid: continue

        
        if has_search:
                
                if '</search>' in area:
                    
                    actual_q = area.split('<search>')[-1].split('</search>')[0].strip()
                else:
                    
                    actual_q = area.split('<search>')[-1].strip()

                
                if len(actual_q) < 5:
                    sub_scores['reward/format_correctness'] -= 0.1 
                else:
                    
                    prev_c = ref_trace[i-1].get('expected_content', "") if i > 0 else ""
                    #
                    target_query = problem_text + " " + prev_c
                    
                    q_sim = calculate_similarity(actual_q, target_query)
                    
                    sub_scores['reward/query_concatenation'] += (q_sim * 0.4 / num_trace)

        except: 
            pass

        
        content_sim = calculate_similarity(clean_content, ref.get('expected_content', ""))
        sub_scores['reward/content_similarity'] += (content_sim * 0.5 / num_trace)

        
        if has_search and need_search:
            golden_ex = ref.get('golden_example', "")
            if golden_ex:
                u_sim = calculate_similarity(clean_content, golden_ex)
                if u_sim > 0.3: 
                    sub_scores['reward/retrieval_utilization'] += (retrieval_score / num_trace)

   
    pred_ans = extract_model_answer(solution_str)
    
    
    clean_gt = clean_ground_truth_answer(final_answer_gt)
    
    
    is_correct = False
    if pred_ans and clean_gt:
       
        if normalize_answer(pred_ans) == normalize_answer(clean_gt):
            is_correct = True
    
    if is_correct:
        sub_scores['reward/exact_match'] = 1.0
        total_reward += final_format_score



    total_reward += (
        sub_scores['reward/format_correctness'] +
        sub_scores['reward/decision_accuracy'] +
        sub_scores['reward/content_similarity'] +
        sub_scores['reward/query_concatenation'] +
        sub_scores['reward/retrieval_utilization']
    )
    
    sub_scores['reward/total_reward'] = total_reward
    return total_reward, sub_scores