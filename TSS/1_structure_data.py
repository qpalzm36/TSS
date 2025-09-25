import openai
import os
import json
from tqdm import tqdm

# --- 配置 ---
os.environ["OPENAI_API_KEY"] = ""
try:
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="")
except TypeError:
    print("")
    exit(1)

MODEL = "gpt-4o-mini"

# ---  ---
RAW_RETRIEVER_DATA_PATH = ""
STRUCTURED_RETRIEVER_DATA_PATH = ""

RAW_GENERATOR_DATA_PATH = ""
STRUCTURED_GENERATOR_DATA_PATH = ""

RAW_KNOWLEDGE_BASE_PATH = ""
STRUCTURED_KNOWLEDGE_BASE_PATH = ""

ERROR_LOG_FILE = ""

# --- - ---
SYSTEM_PROMPT = """
You are an expert in mathematical reasoning. Your task is to break down a solution to a math problem into a series of logical steps. Each step must follow this strict format: "[CONDITION] ... [PROCESS] ... [CONCLUSION] ..."

**Format Definition:**
- [CONDITION]: What is known at the beginning of this step? This must include all relevant information from the initial problem statement AND the conclusions from any preceding steps that are required for the current process.
- [PROCESS]: The core reasoning or calculation performed. Be concise but maintain logical and mathematical rigor.
- [CONCLUSION]: The result or new information derived at the end of this step.

**Language Requirement:**
- ALL responses must be in English, including all text descriptions
- Only mathematical expressions should remain in their original format (LaTeX)

**Mathematical Notation Requirements:**
- All mathematical formulas, equations, variables, and expressions MUST be formatted in LaTeX
- Use \\( ... \\) for inline math (e.g., \\( x^2 + y^2 = 10 \\))
- Use \\[ ... \\] for display math when needed
- Include ALL mathematical symbols, variables, numbers in mathematical context within LaTeX formatting
- Examples: \\( x = 2 \\), \\( y = -3x + 2 \\), \\( \\frac{a}{b} \\), \\( \\sqrt{c} \\)

**Conciseness Requirement:**
- Use minimal natural language while preserving essential meaning and logical coherence
- Focus on mathematical content rather than verbose explanations
- Eliminate unnecessary descriptive words but keep mathematical precision

**Example:**
Input: {"problem": "Let $g_0(x) = x + |x-200|-|x+200|$, and for $n \geq 1$, let $g_n(x) = |g_{n-1}(x)|-1$. For how many values of $x$ is $g_{150}(x)=0$?", "solution": "First, simplify $g_0(x)$:\n\\[ g_0(x) = \\left\\{\n\\begin{array}{cl}\nx + 400 & \\text{if } x < -200, \\\\\n-x & \\text{if } -200 \\le x < 200, \\\\\nx - 400 & \\text{if } x \\ge 200.\n\\end{array}\n\\right. \\]\n\nWe know that $g_n(x) = |g_{n-1}(x)| - 1$, so:\n1. If $g_{n-1}(x) = k$, then $g_n(x) = k - 1.$\n2. Specifically, if $g_{149}(x) = 1$ and $g_{150}(x) = 0$ then $g_0(x) = 150$ or $-150$.\n\nAnalyze solutions based on the piecewise values of $g_0(x)$:\n- **Case 1: $x+400 = 150 \\Rightarrow x = -250$ and $x < -200$, true.**\n- **Case 2: $-x = 150 \\Rightarrow x = -150$ and $-200 \\leq x < 200$, true.**\n- **Case 3: $x - 400 = 150 \\Rightarrow x = 550$ and $x \\geq 200$, true.**\n\nSimilar calculations result in values $x = -150, 150$ being solutions too. Hence, each processed equation of $g_0(x) = \\pm 150$ has one solution.\n\nConclusion:\n\\[ \\boxed{4} \\]"}

Output: {
  "problem": "Let \\(g_0(x) = x + |x-200|-|x+200|\\), and for \\(n \\geq 1\\), let \\(g_n(x) = |g_{n-1}(x)|-1\\). For how many values of \\(x\\) is \\(g_{150}(x)=0\\)?",
  "Step 1": "[CONDITION] The function is defined as \\(g_0(x) = x + |x-200| - |x+200|\\). [PROCESS] Simplify the expression by analyzing the absolute values over three intervals: \\(x < -200\\), \\(-200 \\le x < 200\\), and \\(x \\ge 200\\). [CONCLUSION] The piecewise definition of the function is \\( g_0(x) = \\begin{cases} x+400 & \\text{if } x < -200 \\\\ -x & \\text{if } -200 \\le x < 200 \\\\ x-400 & \\text{if } x \\ge 200 \\end{cases} \\).",
  "Step 2": "[CONDITION] The target is \\(g_{150}(x) = 0\\) and the recurrence relation is \\(g_n(x) = |g_{n-1}(x)| - 1\\). [PROCESS] Work backward from \\(n=150\\). \\(g_{150}(x) = 0 \\implies |g_{149}(x)| - 1 = 0 \\implies |g_{149}(x)| = 1\\). Applying the relation backwards \\(150\\) times, \\(|g_0(x)| = |g_{149}(x)| + 149 = 1 + 149 = 150\\). [CONCLUSION] The problem reduces to solving \\(g_0(x) = 150\\) or \\(g_0(x) = -150\\).",
  "Step 3": "[CONDITION] The piecewise definition of \\(g_0(x)\\) and the equation \\(g_0(x) = 150\\). [PROCESS] Solve for \\(x\\) in each piece: 1) \\(x+400=150\\) for \\(x<-200\\), 2) \\(-x=150\\) for \\(-200 \\le x < 200\\), 3) \\(x-400=150\\) for \\(x \\ge 200\\). [CONCLUSION] The valid solutions for this case are \\(x = -250\\), \\(x = -150\\), and \\(x = 550\\).",
  "Step 4": "[CONDITION] The piecewise definition of \\(g_0(x)\\) and the equation \\(g_0(x) = -150\\). [PROCESS] Solve for \\(x\\) in each piece: 1) \\(x+400=-150\\) for \\(x<-200\\), 2) \\(-x=-150\\) for \\(-200 \\le x < 200\\), 3) \\(x-400=-150\\) for \\(x \\ge 200\\). [CONCLUSION] The valid solutions for this case are \\(x = -550\\), \\(x = 150\\), and \\(x = 250\\).",
  "Step 5": "[CONDITION] The set of solutions from \\(g_0(x)=150\\) is \\(\\{-250, -150, 550\\}\\). The set of solutions from \\(g_0(x)=-150\\) is \\(\\{-550, 150, 250\\}\\). [PROCESS] Combine the two sets of solutions and count the number of unique elements. [CONCLUSION] There are a total of \\(6\\) unique values for \\(x\\)."
}

**Instructions:**
1. You will receive a JSON object with "problem" and "solution" fields
2. Break down the "solution" into logical steps following the [CONDITION][PROCESS][CONCLUSION] format
3. Return a JSON object with "problem" field and step fields like "Step 1", "Step 2", etc.
4. Each step value should be a concise string in the format "[CONDITION] ... [PROCESS] ... [CONCLUSION] ..."
5. ALL mathematical content must be in LaTeX format using \\( ... \\) or \\[ ... \\]
6. Minimize natural language while preserving mathematical meaning and logical flow
7. Ensure all JSON strings are properly escaped
"""

def call_gpt4_for_structuring(problem, solution):
    """"""
    user_input = {
        "problem": problem,
        "solution": solution
    }

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=prompt_messages,
            temperature=0.1,
            max_tokens=2048,
            top_p=0.9,
            stream=False,
            response_format={"type": "json_object"}
        )
        gpt_response_content = response.choices[0].message.content
        return json.loads(gpt_response_content), None
    except json.JSONDecodeError as e:
        error_info = {
            "error_type": "JSONDecodeError",
            "message": str(e),
            "gpt_response": gpt_response_content,
            "problem": problem,
            "solution": solution
        }
        return None, error_info
    except Exception as e:
        error_info = {
            "error_type": "UnknownError", 
            "message": str(e),
            "problem": problem,
            "solution": solution
        }
        return None, error_info

def process_line(line):

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        print(f" {line.strip()}")
        return None, None

    problem = data.get("problem")
    solution = data.get("solution")

    if not problem or not solution:
        return None, None

    structured_result, error = call_gpt4_for_structuring(problem, solution)
    return structured_result, error

def process_file(input_path, output_path):

    print(f"{input_path}")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    error_dir = os.path.dirname(ERROR_LOG_FILE)
    if error_dir:
        os.makedirs(error_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile, \
         open(ERROR_LOG_FILE, 'a', encoding='utf-8') as error_file:

        lines = infile.readlines()
        success_count = 0
        error_count = 0
        
        for line in tqdm(lines, desc=f"结构化 {os.path.basename(input_path)}"):
            result, error = process_line(line)
            if result:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                outfile.flush()
                success_count += 1
            elif error:
                error_file.write(f"文件: {input_path}\n")
                error_file.write(json.dumps(error, ensure_ascii=False) + '\n')
                error_file.flush()
                error_count += 1

    print(f" {os.path.basename(input_path)}")
    print(f" {success_count}, : {error_count}")
    print(f" {output_path}")
    
def main():
   
    if not os.environ.get("OPENAI_API_KEY"):
        print("")
        return

    files_to_process = [
        (RAW_RETRIEVER_DATA_PATH, STRUCTURED_RETRIEVER_DATA_PATH),
        (RAW_KNOWLEDGE_BASE_PATH, STRUCTURED_KNOWLEDGE_BASE_PATH),
        (RAW_GENERATOR_DATA_PATH, STRUCTURED_GENERATOR_DATA_PATH)
    ]
    """ files_to_process = [
        (RAW_GENERATOR_DATA_PATH, STRUCTURED_GENERATOR_DATA_PATH)
    ] """
    for input_path, output_path in files_to_process:
        if os.path.exists(input_path):
            process_file(input_path, output_path)
        else:
            print(f"{input_path}")

    print("！")
    print(f": {ERROR_LOG_FILE}")

if __name__ == "__main__":
    main()
