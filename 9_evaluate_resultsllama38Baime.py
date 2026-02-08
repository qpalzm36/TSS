# 修改后的 9_evaluate_resultsllama38Baime_aime.py

import os
import json
import openai
from tqdm import tqdm
import argparse  


os.environ["OPENAI_API_KEY"] = ""
API_BASE_URL = ""

EVALUATION_MODEL = "gpt-4o-mini"


PROMPT_TEMPLATES = {
    "answer_accuracy": {
        "system": "You are a mathematical answer comparison expert. Your task is to determine if an AI-generated answer (Generated Answer) is mathematically equivalent to the standard answer (Gold Answer). Ignore differences in units, formatting, or trailing zeros. For example, '12' and '\\boxed{12}' are equivalent. For AIME problems, answers are typically integers from 0-999. Return a JSON object containing only one key, 'is_correct', with a value of true or false.",
        "user": lambda gold, gen: {"gold_answer": str(gold), "generated_answer": str(gen)}
    }
}



def get_openai_client():

    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=API_BASE_URL)
        return client
    except Exception as e:

        return None

def call_evaluator(client, metric_name, **kwargs):

    if not client: return None
    
    template = PROMPT_TEMPLATES[metric_name]
    user_content = template["user"](**kwargs)

    try:
        response = client.chat.completions.create(
            model=EVALUATION_MODEL,
            messages=[
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        gpt_response_content = response.choices[0].message.content
        return json.loads(gpt_response_content)
    except Exception as e:

        return None

def main(args):

    client = get_openai_client()
    if not client:

        return

    if not os.path.exists(args.inference_log_path):

        return

    all_results = []
    with open(args.inference_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_results.append(json.loads(line))

    full_evaluation_data = []
    
    for result in tqdm(all_results, desc=""):
        problem_eval = {
            "problem_id": result["problem_id"],
            "problem_text": result["problem_text"],
            "gold_answer": result["gold_answer"],
            "generated_answer": result["final_generated_answer"],
            "is_correct": None
        }


        if result["gold_answer"] is not None and result["final_generated_answer"] is not None:
            eval_res = call_evaluator(client, "answer_accuracy", gold=result["gold_answer"], gen=result["final_generated_answer"])
            if eval_res and isinstance(eval_res.get("is_correct"), bool):
                problem_eval["is_correct"] = eval_res["is_correct"]
        
        full_evaluation_data.append(problem_eval)


    total_problems = len(full_evaluation_data)
    final_answer_correct_count = sum(1 for r in full_evaluation_data if r["is_correct"] is True)
    
    report = {
        "total_problems_evaluated": total_problems,
        "final_answer_accuracy": f"{final_answer_correct_count / total_problems:.2%}" if total_problems > 0 else "N/A",
        "detailed_results": full_evaluation_data
    }



    with open(args.evaluation_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--inference_log_path", type=str, default="]]", help="]")
    parser.add_argument("--evaluation_report_path", type=str, default="]", help="]")
    args = parser.parse_args()
    main(args)
