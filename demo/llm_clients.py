import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class HelloAgentsLLM:
    """Initialize an OpenAI-compatible LLM client.

    Reads defaults from env vars:
    - `LLM_MODEL_ID`
    - `LLM_API_KEY`
    - `LLM_BASE_URL`
    - `LLM_TIMEOUT` (seconds, default: 60)

    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.model = model or api_model or os.getenv("LLM_MODEL_ID")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))

        missing: List[str] = []
        if not self.model:
            missing.append("LLM_MODEL_ID")
        if not api_key:
            missing.append("LLM_API_KEY")
        if not base_url:
            missing.append("LLM_BASE_URL")
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            collected_events: List[str] = []
            for chunk in response:
                content = (chunk.choices[0].delta.content or "")
                if content:
                    print(content, end="", flush=True)
                    collected_events.append(content)

            print()
            return "".join(collected_events)
        except Exception as e:
            print(f"call llm api error: {e}")
            return None
