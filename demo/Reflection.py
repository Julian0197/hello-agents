from llm_clients import HelloAgentsLLM
from typing import List, Dict, Any

# Memory Module

class Memory:
  """
  A simple short-term memory module to store agent' s execution and reflection get_trajectory.
  """

  def __init__(self):
    self.records: List[Dict[str, Any]] = []

  def add_record(self, record_type: str, content: str):
    """
    Add a new record to the memory

    Parameters:
    - record_type(str): The type of the record ('execution' or 'reflection')
    - content(str): The content of the record(e.g., generated code or reflection feedback)
    """

    self.records.append({"type": record_type, "content": content})
    print(f"Memory updated, added one {record_type} record")

  def add_trajectory(self) -> str:
    """
    Format all memory records into a coherent text string for prompt construction. 
    """
    trajectory = ""
    for record in self.records:
      if record['type'] == 'execution':
        trajectory += f"--- Previous attempt(code) ---\n {record['content']}\n\n"
      elif record['type'] == 'reflection':
        trajectory += f"--- Reviewer feedback ---\n {record['content']}\n\n"
    return trajectory.strip()
    
  def get_last_execution(self) -> str:
    """
    Get the most recent execution result(e.g., the latest generated code)
    """
    for record in reversed(self.records):
      if record["type"] == "execution":
        return record["content"]
    return None
  
# initial prompt
INITIAL_PROMPT_TEMPLATE = """
You are a professional Python developer.Please write a Python function according to
the following requirements.Your code must include a complete function signature, docstring,
and comply with the PEP 8 coding format.

# Task:
{task}

Please output the code directly, do not include extra explanation.
"""

# reflection prompt
REFLECTION_PROMPT_TEMPLATE = """
You are an extremely strict code review expert and a senior algorithm engineer with high
requirements for code performance.

# Task:
{task}

# Code to be reviewed:
```
{code}
```

Please analyze the time complexity of the code and consider whether there is a more
algorithmically efficient solution that can significantly improve performance.If it
exists, clear points the deficiencies of the current algorithm and provide specific
and feasible suggestions.Only if the code is already optimal at the algorithm level, 
output: "No need to improve".

Please output your feedback directly, do not include any extra explanation.
"""

# refinement prompt
REFINEMENT_PROMPT_TEMPLATE = """
You are a sophisticated Python programmer.Please optimize your code base on the feedback
from the code review expert.

# Task:
{task}

# Last code attempt:
{last_code_attempt}

## code reviewer feedback
{feedback}

Please generate a improved new version of the code according to thr reviewer' s feedback.
You code must include complete function signature, docstring and follow PEP 8 coding format.
Please directly output the optimized code, do not include any extra explanation.
"""

class ReflectionAgent:
  def __init__(self, llm_client, max_iteration = 3):
    self.llm_client = llm_client
    self.memory = Memory()
    self.max_iteration = max_iteration

  def run(self, task: str):
    print(f"\n--- Starting the task --- \nTask: {task}")

    # init
    print("\n --- Performing the initial attempt ---")
    initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
    initial_code = self._get_llm_response(prompt=initial_prompt)
    self.memory.add_record("execution", initial_code)

    # iteration
    for i in range(self.max_iteration):
      print(f"\n --- Iteration {i+1}/{self.max_iteration} ---")

      # reflection
      print("\n -> Reflecting...")
      last_code = self.memory.get_last_execution()
      reflect_prompt = REFLECTION_PROMPT_TEMPLATE.format(task=task, code=last_code)
      feedback = self._get_llm_response(prompt=reflect_prompt)
      self.memory.add_record(record_type="reflection", content=feedback)

      # Check if iteration needs to terminate
      if "无需改进" or "no need to improve" in feedback.lower():
        print("\n✅Task completed, no further improvement is needed.")
        break

      # refinement
      print("\n-> Refining...")
      refine_prompt=REFINEMENT_PROMPT_TEMPLATE.format(
        task=task,
        last_code_attempt=last_code,
        feedback=feedback
      )
      refine_code = self._get_llm_response(refine_prompt)
      self.memory.add_record(record_type="execution", content=refine_code)
    
    final_code = self.memory.get_last_execution()
    print(f"\n --- Task completed --- \nFinal generated code:\n{final_code}")
    return final_code

  def _get_llm_response(self, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    response_text = self.llm_client.think(messages=messages) or ""
    return response_text

if __name__ == '__main__':
  llm_client = HelloAgentsLLM()
  agent = ReflectionAgent(llm_client, max_iteration=3)
  task = "Write a python function: find all prime number between 1 and N"
  agent.run(task)

