"""
手势词序列 → 自然语言润色（Qwen / OpenAI / 演示模式）

Qwen 调用位置：本文件的 `_polish_via_qwen()`，通过阿里云 DashScope 兼容 OpenAI 的接口。

配置（任选其一）:
  set DASHSCOPE_API_KEY=sk-xxx        # 推荐，Qwen
  set OPENAI_API_KEY=sk-xxx           # 备选

无 API Key 时自动走演示规则，便于答辩展示。

示例:
  python gesture_polish.py
  >>> polish_words(["我", "医院", "帮助"])
  "我需要去医院，希望得到帮助。"
"""

from __future__ import annotations

import env_loader  # noqa: F401 — 加载 .env 中的 DASHSCOPE_API_KEY

import json
import os
import urllib.error
import urllib.request
from typing import Literal

Provider = Literal["qwen", "openai", "demo"]

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-turbo").strip() or "qwen-turbo"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
API_TIMEOUT_SEC = float(os.environ.get("POLISH_TIMEOUT_SEC", "6"))

_SYSTEM_PROMPT = (
    "你是手语转译助手。用户通过手势依次表达了若干词语，"
    "请将其润色为一句自然、流畅的中文。"
    "不要添加词语中没有的信息，语气简洁、礼貌。"
    "只输出一句话，不要解释、不要引号。"
)


def _detect_provider() -> Provider:
    if os.environ.get("DASHSCOPE_API_KEY", "").strip():
        return "qwen"
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    return "demo"


def provider_label() -> str:
    p = _detect_provider()
    return {"qwen": "Qwen API", "openai": "OpenAI API", "demo": "演示模式（无 API Key）"}[p]


def get_polish_info() -> dict:
    p = _detect_provider()
    model = None
    if p == "qwen":
        model = QWEN_MODEL
    elif p == "openai":
        model = os.environ.get("OPENAI_MODEL", OPENAI_MODEL).strip() or OPENAI_MODEL
    return {
        "provider": p,
        "label": provider_label(),
        "model": model,
    }


def _build_user_prompt(words: list[str]) -> str:
    joined = "、".join(words)
    return f"词语：{joined}"


def _chat_completion(*, base_url: str, api_key: str, model: str, words: list[str]) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(words)},
        ],
        "temperature": 0.2,
        "max_tokens": 80,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=API_TIMEOUT_SEC) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"].strip()
    return _strip_wrapping_quotes(content)


def _strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    for pair in ('"""', "'''", '"', "'", "「", "」", "“", "”"):
        if text.startswith(pair) and text.endswith(pair) and len(text) > len(pair) * 2:
            return text[len(pair) : -len(pair)].strip()
    return text


def polish_danger_hospital(words: list[str]) -> str | None:
    """
    危险 + 医院 → 紧急就医句式。
    若同时出现「谢谢」，句末追加「非常感谢」。
    """
    s = {w.strip() for w in words if w and w.strip()}
    if not {"危险", "医院"} <= s:
        return None
    if "谢谢" in s:
        return "我遇到了危险，麻烦送我去医院，非常感谢。"
    return "我遇到了危险，麻烦送我去医院。"


def _polish_via_qwen(words: list[str]) -> str:
    """Qwen 调用入口 — 使用 DashScope API Key。"""
    api_key = os.environ["DASHSCOPE_API_KEY"].strip()
    return _chat_completion(
        base_url=QWEN_BASE_URL,
        api_key=api_key,
        model=QWEN_MODEL,
        words=words,
    )


def _polish_via_openai(words: list[str]) -> str:
    api_key = os.environ["OPENAI_API_KEY"].strip()
    base = os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL).strip() or OPENAI_BASE_URL
    model = os.environ.get("OPENAI_MODEL", OPENAI_MODEL).strip() or OPENAI_MODEL
    return _chat_completion(base_url=base, api_key=api_key, model=model, words=words)


def _polish_demo(words: list[str]) -> str:
    """无 API 时的演示润色（覆盖常见答辩场景）。"""
    single_map = {
        "你好": "你好，很高兴见到你",
        "帮助": "我需要帮助",
        "医院": "我需要去医院",
        "谢谢": "谢谢你",
        "危险": "我有危险，请帮帮我",
        "我": "是我",
    }
    if len(words) == 1 and words[0] in single_map:
        return single_map[words[0]]

    s = set(words)

    if s >= {"我", "医院", "帮助"}:
        return "我需要去医院，希望得到帮助。"
    if s >= {"我", "危险", "帮助"}:
        return "我遇到了危险，请帮助我！"
    if s >= {"我", "医院"}:
        return "我需要去医院。"
    if s >= {"我", "帮助"}:
        return "我需要帮助。"
    if s >= {"我", "危险"}:
        return "我有危险，请帮帮我！"
    if "你好" in s and "谢谢" in s:
        return "你好，非常感谢你的帮助。"
    if words == ["你好"]:
        return "你好，很高兴见到你"
    if words == ["谢谢"]:
        return "谢谢你"

    if len(words) == 1:
        w = words[0]
        if w == "我":
            return "是我"
        if w in single_map:
            return single_map[w]
        return w

    if len(words) == 2:
        a, b = words[0], words[1]
        pair_rules = {
            ("我", "医院"): "我需要去医院。",
            ("我", "帮助"): "我需要帮助。",
            ("我", "危险"): "我有危险，请帮帮我！",
            ("你好", "谢谢"): "你好，非常感谢。",
            ("谢谢", "帮助"): "谢谢你的帮助。",
            ("医院", "帮助"): "请帮我去医院。",
        }
        if (a, b) in pair_rules:
            return pair_rules[(a, b)]
        if (b, a) in pair_rules:
            return pair_rules[(b, a)]

    return "，".join(words) + "。"


def polish_words(
    words: list[str],
    *,
    provider: Provider | None = None,
) -> str:
    """
    将手势识别词序列润色为自然语句。

    Args:
        words: 中文词列表，如 ["我", "医院", "帮助"]
        provider: 强制指定 qwen / openai / demo；默认自动检测

    Returns:
        润色后的完整句子
    """
    cleaned = [w.strip() for w in words if w and w.strip()]
    if not cleaned:
        return ""

    emergency = polish_danger_hospital(cleaned)
    if emergency:
        return emergency

    use = provider or _detect_provider()

    if use == "demo":
        return _polish_demo(cleaned)

    try:
        if use == "qwen":
            return _polish_via_qwen(cleaned)
        if use == "openai":
            return _polish_via_openai(cleaned)
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError) as exc:
        print(f"[润色] API 超时或失败，使用本地快速润色: {exc}")
    except Exception as exc:
        print(f"[润色] 未知错误，使用本地快速润色: {exc}")

    return _polish_demo(cleaned)


def polish_gesture(zh: str) -> str:
    """将单个手势词润色为自然语句（交流记录自动调用）。"""
    zh = zh.strip()
    if not zh:
        return ""
    return polish_words([zh])


def words_from_history(history: list[dict]) -> list[str]:
    """从历史记录提取按时间顺序的去重词语（history 为新→旧）。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in reversed(history):
        zh = str(item.get("zh", "")).strip()
        if zh and zh not in seen:
            seen.add(zh)
            ordered.append(zh)
    return ordered


if __name__ == "__main__":
    samples = [
        ["危险", "医院"],
        ["危险", "医院", "谢谢"],
        ["我", "危险", "医院", "谢谢"],
        ["我", "医院", "帮助"],
        ["你好", "谢谢"],
        ["我", "危险"],
        ["帮助"],
    ]
    print(f"当前模式: {provider_label()}\n")
    for words in samples:
        print(f"输入: {words}")
        print(f"输出: {polish_words(words)}\n")
