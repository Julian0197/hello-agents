import ast
from llm_clients import HelloAgentsLLM
from typing import List
from dotenv import load_dotenv

load_dotenv()

PLANNER_PROMPT_TEMPLATE = """
You are a top-tier AI planning expert.Your task is to break down complex questions
raised by users into a series of simple step-by-step tasks.Ensure that each task
in the plan is an independent, executable subtask, and arrange them in strict logical order.

Question: {question}

Please output your plan strictly in the following format, ```python prefix and ``` suffix is necessary:
```python
['step1', 'step2', 'step3', ...]
```
"""

class Planner:
  def __init__(self, llm_client: HelloAgentsLLM):
    self.llm_client = llm_client

  def plan(self, question: str) -> list[str]:
    prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
    messages = [{'role': 'user', 'content': prompt}]

    print("--- Executing the plan... ---")
    response_text = self.llm_client.think(messages=messages) or ""
    print(f"Plan finished: \n{response_text}")

    try:
      plan_str = response_text.split("```python")[1].split("```")[0].strip()
      plan = ast.literal_eval(plan_str)
      return plan if isinstance(plan, list) else []
    except(ValueError, SyntaxError, IndexError) as e:
      print(f"Parsing plan error: {e}")
      print(f"Original response: {response_text}")
      return
    

EXECUTOR_PROMPT_TEMPLATE = """
You are a professional AI execution expert.Your mission is to strictly follow the
given plan and solve the problem step by step.You will receive the original question,
the full plan and the completed plan result so far.Please focus on solving the 
current step and output the final answer for current step.Do not output any extra
explanation or conversations.

# Original question:
{question}

# Full plan
{plan}

# History steps and results
{history}

# Current step:
{current_step}

Only output the current step!
"""

class Executor:
  def __init__(self, llm_client: HelloAgentsLLM):
    self.llm_client = llm_client
  
  def execute(self, question: str, plan: list[str]) -> str:
    history = ""
    final_answer = ""

    print("\n--- Executing the plan ---")
    for i, step in enumerate(plan, 1):
      print(f"\n Executing progress {i}/{len(plan)}: {step}")
      prompt = EXECUTOR_PROMPT_TEMPLATE.format(
        question=question, 
        plan=plan, 
        history=history if history else 'Empty',
        current_step=step
      )
      messages = [{"role": "user", "content": prompt}]
      response_text = self.llm_client.think(messages=messages)

      history += f"Step {i}: {step}\nResult: {response_text}\n\n"
      final_answer = response_text
      print(f"✅Step {i} completed, result: {final_answer}")
    
    return final_answer

class PlanAndSolveAgent:
  def __init__(self, llm_client: HelloAgentsLLM):
    self.llm_client = llm_client
    self.planner = Planner(self.llm_client)
    self.executor = Executor(self.llm_client)

  def run(self, question: str):
    print(f"\n--- Start to plan and solve question: {question} --- ")
    plan = self.planner.plan(question=question)
    if not plan:
      print("\n --- Mission terminated \nCannot generate valid plan ---")
      return
    final_answer = self.executor.execute(question=question, plan=plan)
    print(f"\n --- Mission completed --- \n 🔚Final answer: {final_answer} ")

  
if __name__ == '__main__':
  try:
    llm_client = HelloAgentsLLM()
    agent = PlanAndSolveAgent(llm_client)
    question = "帮我写一个 Python 函数，输入一个列表，返回其中出现次数最多的元素和出现次数"
    agent.run(question)
  except ValueError as e:
    print(e)