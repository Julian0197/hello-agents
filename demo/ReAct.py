from llm_clients import HelloAgentsLLM
from tools import ToolsExecutor, search
import re


# ReAct: Reasoning and Acting Prompt
REACT_PROMPT_TEMPLATE = """
You are a helpful assistant that can use external tools to answer questions.
The available tools are:
{tools}

Please response strictly following the format:
Thought: Your reasoning process, used to analyze the question, break down the problem, and plan the next step.
Action: The action you decide to take, it must be in one of the formats:
- `{{tool_name}}[{{tool_input}}]`: call an available tool
- `Finish[{{final_answer}}]`: finish the task when you think the answer is ready

Now, let's think step by step to solve the question.
Question: {question}
History: {history}
"""


# question => [start loop] => combination Prompt with history => LLM => parse Thought/Action
# => actions => get Observation => update History => [next loop] => ... => Finish
class ReActAgent:
  def __init__(self, llm_client, tool_executor, max_steps: int = 5):
    self.llm_client = llm_client
    self.tool_executor = tool_executor
    self.max_steps = max_steps
    self.history = []
  
  def run(self, question: str):
    current_step = 0

    while current_step < self.max_steps:
      # combine prompts wite history and tools desc
      current_step += 1
      print(f"\n --- Step {current_step} ---")
      tools_desc = self.tool_executor.getAvailableTools()
      history_str = '\n'.join(self.history)
      prompt = REACT_PROMPT_TEMPLATE.format(tools = tools_desc, question = question, history = history_str)
      
      # call llm
      messages = [{"role": "user", "content": prompt}]
      response_text = self.llm_client.think(messages = messages, temperature = 0.5)
      if not response_text:
        print("Error: cannot get valid response from LLM") 
        break

      thought, action = self._parse_output(response_text)
      if thought: print(f"🤔 Thought: {thought}")
      if not action:
        print("Warning: fail to parse valid Action, process terminated.")
        break
      if action.startswith("Finish"):
        final_answer = self._parse_action_input(action)
        print(f"🎉Final answer: {final_answer}")
        return final_answer
      
      # use tool
      tool_name, tool_input = self._parse_action(action)
      if not tool_name or not tool_input:
        self.history.append("Observation: invalid Action format, please check.")
        continue
      print(f"🚶 Parsed action: {tool_name}[{tool_input}]")
      tool_function = self.tool_executor.getTool(tool_name)
      observation = tool_function(tool_input) if tool_function else f"Error: cannot find '{tool_name}' tool"
      print(f"👁Observation: {observation}")
      self.history.append(f"Action: {action}")
      self.history.append(f"Observation: {observation}")
    
    print("Maximum steps reached, process stopped")
    return None
  
  # parse Thought and Action from llm output
  def _parse_output(self, text: str):
    # Thought: Match to "Action:" or the end of the text
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
    # Action: Match to the end of the text
    action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else None
    action = action_match.group(1).strip() if action_match else None
    return thought, action
  
  def _parse_action(self, action_text: str):
      if not action_text:
        return (None, None)

      s = action_text.strip()
      # allow wrapping in backticks
      if s.startswith("`") and s.endswith("`"):
        s = s.strip("`").strip()

      # Preferred format: ToolName[tool_input]
      match = re.match(r"^([A-Za-z_]\w*)\[(.*)\]$", s, re.DOTALL)
      if match:
        return (match.group(1), match.group(2).strip())

      # Compatibility format (common model output): ToolName: tool_input
      match = re.match(r"^([A-Za-z_]\w*)\s*:\s*(.*)$", s, re.DOTALL)
      if match:
        return (match.group(1), match.group(2).strip())

      return (None, None)

  def _parse_action_input(self, action_text: str):
      if not action_text:
        return ""

      s = action_text.strip()
      if s.startswith("`") and s.endswith("`"):
        s = s.strip("`").strip()

      # Finish[final_answer]
      match = re.match(r"^Finish\[(.*)\]$", s, re.DOTALL)
      if match:
        return match.group(1).strip()

      # Finish: final_answer (compat)
      match = re.match(r"^Finish\s*:\s*(.*)$", s, re.DOTALL)
      if match:
        return match.group(1).strip()

      # ToolName[tool_input]
      match = re.match(r"^[A-Za-z_]\w*\[(.*)\]$", s, re.DOTALL)
      return match.group(1).strip() if match else ""


if __name__ == '__main__':
  llm = HelloAgentsLLM()
  tool_executor = ToolsExecutor()
  search_desc = "A browser search engine, when you want to acquire external knowledge, such as: recent events, real-time data, factual information."
  tool_executor.registerTool("Search", search_desc, search)
  agent = ReActAgent(llm, tool_executor=tool_executor)
  question = "Combined with today' s weather in Shanghai, what outdoor sports do you recommend?"
  agent.run(question)
