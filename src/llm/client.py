from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    api_base: str
    api_key: str
    model: str
    timeout_seconds: int = 60
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_base = os.getenv("LLM_API_BASE", "").rstrip("/")
        api_key = os.getenv("LLM_API_KEY", "")
        model = os.getenv("LLM_MODEL", "")
        timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        if not api_base or not api_key or not model:
            raise ValueError("Missing LLM_API_BASE, LLM_API_KEY, or LLM_MODEL in environment.")
        return cls(
            api_base=api_base,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )

class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        url = f"{self.config.api_base}/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if "text" in part:
                        text_parts.append(part["text"])
                    elif part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
            content = "".join(text_parts)

        if not content or not str(content).strip():
            raise ValueError(f"LLM returned empty content: {data}")

        content = str(content).strip()

        # 去掉 ```json ... ``` 包裹
        if content.startswith("```"):
            content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"^```\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        # 记录 LLM 原始输出到文件
        self._log_response(content)

        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 提取第一个完整 {...} JSON 块（按括号深度匹配）
        start = content.find("{")
        if start != -1:
            depth = 0
            in_string = False
            escape_next = False
            end = -1
            for i in range(start, len(content)):
                ch = content[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    pass

        # 回退：贪婪正则
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"LLM did not return valid JSON. Raw content:\n{content}")

    def _log_response(self, content: str) -> None:
        """将 LLM 原始响应追加写入日志文件。"""
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        log_path = log_dir / f"llm_responses_{today}.log"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"\n{'='*60}\n[{timestamp}] Model: {self.config.model}\n{'='*60}\n{content}\n"

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            logger.warning("Failed to write LLM response log: %s", e)

