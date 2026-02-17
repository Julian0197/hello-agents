import os
from typing import Any, Callable, Dict

from dotenv import load_dotenv

load_dotenv()


def _serpapi_get_dict(params: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """call SerpAPI (Google) with params and return a dict."""
    try:
        from serpapi import SerpApiClient

        client = SerpApiClient({**params, "api_key": api_key})
        return client.get_dict()
    except Exception:
        pass

    import serpapi

    client = serpapi.Client(api_key=api_key)
    results = client.search(params)
    if hasattr(results, "as_dict"):
        return results.as_dict()
    if isinstance(results, dict):
        return results
    return {"raw": results}


def search(query: str) -> str:
    """call SerpAPI (Google) to search and return a summary."""
    print(f"🔍 calling [SerpApi]: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "SERPAPI_API_KEY not found"

        params: Dict[str, Any] = {
            "engine": "google",
            "q": query,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn",  # 语言代码
        }
        results = _serpapi_get_dict(params, api_key=api_key)

        # 智能解析：优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"sorry, there is no information about '{query}'"
    except Exception as e:
        return f"search error: {e}"


class ToolsExecutor:
    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: Callable[..., Any]) -> None:
        self.tools[name] = {"description": description, "func": func}
        print(f"✅ tool {name} registered")

    def getTool(self, name: str) -> Callable[..., Any] | None:
        return self.tools.get(name, {}).get("func")

    """
    get all available tools description
    """
    def getAvailableTools(self) -> str:
        return "\n".join(
            [
                f"- {name}: {info['description']}"
                for name, info in self.tools.items()
            ]
        )

if __name__ == '__main__':
    # 1. init tool executor
    toolsExecutor = ToolsExecutor()

    # 2. register tools
    toolsExecutor.registerTool(
        "Search",
        "call google search api (SerpApi) to get the answer of the query",
        search,
    )

    # 3. print info
    print("\n ---✅ registered tools ---")
    print(toolsExecutor.getAvailableTools())

    # 4. test action
    print("\n ---✅ test tool call ---")
    tool_name = "Search"
    tool_input = "What is the weather in Shanghai today?"
    tool_function = toolsExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- Observation ---")
        print(observation)
    else:
        print(f"❌ {tool_name} is not registered")
