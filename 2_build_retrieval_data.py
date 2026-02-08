import json
import os
import random
from tqdm import tqdm


STRUCTURED_RETRIEVER_DATA_PATH = ""
RETRIEVER_TRAINING_DATA_PATH = ""

def load_structured_data(file_path):

    if not os.path.exists(file_path):
  
        return []
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
            
    return data

def extract_steps_from_item(item):
    steps = []
    step_num = 1
    while f"Step {step_num}" in item:
        key = f"Step {step_num}"
        value = item[key]
        steps.append((key, value))
        step_num += 1
    return steps



def create_retriever_training_data(structured_data):

    
    if len(structured_data) < 2:
 
        return []

    training_samples = []
    
    for i, p_item in enumerate(tqdm(structured_data, desc="")):
        p_problem = p_item.get("problem", "")
        p_steps = extract_steps_from_item(p_item)
        
        if not p_problem or not p_steps:
            continue
            
 
        query_dict = {"Problem": p_problem}
        query = json.dumps(query_dict, ensure_ascii=False)
        
        p_step_1_key, p_step_1_value = p_steps[0]
        pos_dict = {"Problem": p_problem, p_step_1_key: p_step_1_value}
        positive_passage = json.dumps(pos_dict, ensure_ascii=False)
        
        negatives = []

        if len(p_steps) > 1:
            for future_step_idx in range(1, len(p_steps)):
                future_step_key, future_step_val = p_steps[future_step_idx]
                neg_dict = {"Problem": p_problem, future_step_key: future_step_val}
                negatives.append(json.dumps(neg_dict, ensure_ascii=False))

        q_idx = i
        while q_idx == i:
            q_idx = random.randint(0, len(structured_data) - 1)
        q_item = structured_data[q_idx]
        q_steps = extract_steps_from_item(q_item)
        if q_steps:
            q_step_1_value = q_steps[0][1]
            neg_dict = {"Problem": p_problem, p_step_1_key: q_step_1_value}
            negatives.append(json.dumps(neg_dict, ensure_ascii=False))

        if negatives:
            training_samples.append({"query": query, "pos": [positive_passage], "neg": negatives})


        if len(p_steps) < 2:
            continue

        for p_step_idx in range(len(p_steps) - 1):
            p_current_step_key, p_current_step_value = p_steps[p_step_idx]
            p_next_step_key, p_next_step_value = p_steps[p_step_idx + 1]

            query_dict = {"Problem": p_problem, p_current_step_key: p_current_step_value}
            query = json.dumps(query_dict, ensure_ascii=False)
            
            pos_dict = {"Problem": p_problem, p_current_step_key: p_current_step_value, p_next_step_key: p_next_step_value}
            positive_passage = json.dumps(pos_dict, ensure_ascii=False)
            
            negatives = []

            
            if p_step_idx == 0:
                
                neg_dict_past = {"Problem": p_problem, p_current_step_key: p_current_step_value}
                negatives.append(json.dumps(neg_dict_past, ensure_ascii=False))
            else: # p_step_idx > 0
                p_prev_step_key, p_prev_step_value = p_steps[p_step_idx - 1]
                neg_dict_past = {"Problem": p_problem, p_prev_step_key: p_prev_step_value, p_current_step_key: p_current_step_value}
                negatives.append(json.dumps(neg_dict_past, ensure_ascii=False))

            if len(p_steps) > p_step_idx + 2:
                for future_step_idx in range(p_step_idx + 2, len(p_steps)):
                    p_future_step_key, p_future_step_value = p_steps[future_step_idx]
                    neg_dict = {"Problem": p_problem, p_current_step_key: p_current_step_value, p_future_step_key: p_future_step_value}
                    negatives.append(json.dumps(neg_dict, ensure_ascii=False))

            q_idx = i
            while q_idx == i:
                q_idx = random.randint(0, len(structured_data) - 1)
            q_item = structured_data[q_idx]
            q_problem = q_item.get("problem", "")
            q_steps = extract_steps_from_item(q_item)

            if not q_problem or not q_steps: continue

            q_step_idx = random.randint(0, len(q_steps) - 1)
            _, q_rand_step_value = q_steps[q_step_idx]
            q_rand_next_step_value = ""
            if len(q_steps) > q_step_idx + 1: _, q_rand_next_step_value = q_steps[q_step_idx+1]
                
            negatives.append(json.dumps({"Problem": q_problem, p_current_step_key: p_current_step_value, p_next_step_key: p_next_step_value}, ensure_ascii=False))
            negatives.append(json.dumps({"Problem": p_problem, p_current_step_key: q_rand_step_value, p_next_step_key: p_next_step_value}, ensure_ascii=False))
            if q_rand_next_step_value:
                negatives.append(json.dumps({"Problem": p_problem, p_current_step_key: p_current_step_value, p_next_step_key: q_rand_next_step_value}, ensure_ascii=False))
            if q_rand_next_step_value:
                negatives.append(json.dumps({"Problem": p_problem, p_current_step_key: q_rand_step_value, p_next_step_key: q_rand_next_step_value}, ensure_ascii=False))

            training_samples.append({"query": query, "pos": [positive_passage], "neg": negatives})
            

    return training_samples

def save_training_data(training_data, output_path):

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in training_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main():

    structured_data = load_structured_data(STRUCTURED_RETRIEVER_DATA_PATH)
    if not structured_data:
  
        return

    
    training_data = create_retriever_training_data(structured_data)
    if not training_data:

        return
    
    save_training_data(training_data, RETRIEVER_TRAINING_DATA_PATH)
    

    if training_data:
        sample = training_data[-1]



if __name__ == "__main__":
    main()
