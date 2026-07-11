"""
手势语音播报（本地缓存 + 极速播放）

策略:
  1. 首次运行：用 edge-tts 把 6 个中文词合成到 data/tts_cache/（只需联网一次）
  2. 之后：直接从内存播放缓存，延迟约 50~200ms，不再请求网络

安装:
  pip install edge-tts pygame

可选 API（若以后要更低延迟/定制音色，可自行申请后扩展）:
  - 火山引擎语音合成、阿里云、Azure Speech、讯飞开放平台
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from pathlib import Path

TTS_VOICE = "zh-CN-XiaoxiaoNeural"
CACHE_DIR = Path(__file__).parent / "data" / "tts_cache"

from gesture_custom import load_full_gesture_map

GESTURE_ZH: dict[str, str] = load_full_gesture_map()


def reload_gesture_map() -> None:
    """重新加载内置 + 自定义手势中文映射。"""
    global GESTURE_ZH
    GESTURE_ZH = load_full_gesture_map()

_last_spoken: str | None = None
_lock = threading.Lock()
_speak_queue: queue.Queue[tuple[str, bool] | None] = queue.Queue()
_worker: threading.Thread | None = None
_is_speaking = False
_sounds: dict[str, object] = {}  # zh -> pygame.mixer.Sound
_cache_ready = False
_audio_available = True


def _init_mixer() -> bool:
    """初始化 pygame 音频；云服务器无音频设备时返回 False。"""
    global _audio_available
    if not _audio_available:
        return False
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=256)
        return True
    except Exception as exc:
        _audio_available = False
        print(f"[语音] 无可用音频设备，跳过播报（云部署正常）: {exc}")
        return False


def to_chinese(text: str) -> str:
    key = text.strip().lower()
    return GESTURE_ZH.get(key, text.strip())


def _cache_file(zh_text: str) -> Path:
    return CACHE_DIR / f"{zh_text}.mp3"


def _all_zh_words() -> list[str]:
    return list(dict.fromkeys(GESTURE_ZH.values()))


def is_cache_ready() -> bool:
    return all(_cache_file(zh).exists() for zh in _all_zh_words())


async def _synthesize_one(zh_text: str, mp3_path: Path) -> None:
    import edge_tts

    comm = edge_tts.Communicate(zh_text, TTS_VOICE)
    await comm.save(str(mp3_path))


async def _build_missing_cache() -> list[str]:
    """仅合成缺失的缓存文件，返回新生成的词列表。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [zh for zh in _all_zh_words() if not _cache_file(zh).exists()]
    if not missing:
        return []

    tasks = [_synthesize_one(zh, _cache_file(zh)) for zh in missing]
    await asyncio.gather(*tasks)
    return missing


def build_audio_cache() -> None:
    """生成全部手势语音缓存（首次需联网，约十几秒）。"""
    if is_cache_ready():
        print("[语音] 缓存已存在，跳过合成。")
        return

    print(f"[语音] 正在生成缓存（{len(_all_zh_words())} 个词，需联网）...")
    t0 = time.time()
    created = asyncio.run(_build_missing_cache())
    print(f"[语音] 缓存完成: {created}，耗时 {time.time() - t0:.1f}s")
    print(f"[语音] 目录: {CACHE_DIR}")


def add_tts_for_word(zh_text: str) -> None:
    """为新手势中文词合成并加载 TTS 缓存。"""
    global _cache_ready

    zh_text = zh_text.strip()
    if not zh_text:
        return

    path = _cache_file(zh_text)
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        asyncio.run(_synthesize_one(zh_text, path))
        print(f"[语音] 已合成新手势: {zh_text}")

    if _cache_ready and _audio_available:
        import pygame
        if _init_mixer() and path.exists() and zh_text not in _sounds:
            _sounds[zh_text] = pygame.mixer.Sound(str(path))
            print(f"[语音] 已加载新手势缓存: {zh_text}")


def _load_sounds_to_memory() -> None:
    global _sounds
    if not _init_mixer():
        return

    import pygame

    _sounds.clear()
    for zh in _all_zh_words():
        path = _cache_file(zh)
        if not path.exists():
            raise FileNotFoundError(f"缺少缓存: {path}，请先运行 python gesture_speech.py")
        _sounds[zh] = pygame.mixer.Sound(str(path))
    print(f"[语音] 已加载 {len(_sounds)} 条缓存到内存（即时播放）")


def _play_cached(zh_text: str) -> None:
    if not _audio_available:
        return
    sound = _sounds.get(zh_text)
    if sound is None:
        return
    channel = sound.play()
    if channel is not None:
        while channel.get_busy():
            time.sleep(0.005)


def wait_speech_done(timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _lock:
            speaking = _is_speaking
        if _speak_queue.empty() and not speaking:
            return True
        time.sleep(0.02)
    return False


def _speak_worker() -> None:
    global _last_spoken, _is_speaking

    while True:
        item = _speak_queue.get()
        if item is None:
            break

        text, force = item
        zh_text = to_chinese(text)
        if not zh_text:
            continue

        with _lock:
            if not force and zh_text == _last_spoken:
                continue

        try:
            with _lock:
                _is_speaking = True
            t0 = time.time()
            _play_cached(zh_text)
            elapsed_ms = (time.time() - t0) * 1000
            with _lock:
                _last_spoken = zh_text
                _is_speaking = False
            print(f"[语音] 播报: {zh_text} ({elapsed_ms:.0f}ms)")
        except Exception as exc:
            with _lock:
                _is_speaking = False
            print(f"[语音] 失败: {zh_text} | {exc}")


def _ensure_worker() -> None:
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_speak_worker, name="tts-worker", daemon=True)
        _worker.start()


def init_speech() -> None:
    """准备缓存并启动播报线程（识别程序启动时调用）。"""
    global _cache_ready

    try:
        import edge_tts  # noqa: F401
        import pygame  # noqa: F401
    except ImportError as exc:
        print("[语音] 请安装: pip install edge-tts pygame")
        raise exc

    if not _cache_ready:
        build_audio_cache()
        _load_sounds_to_memory()
        _cache_ready = True

    if _audio_available:
        _ensure_worker()


def get_voice_info() -> str:
    mode = "缓存模式" if is_cache_ready() else "待生成缓存"
    return f"{mode} / {TTS_VOICE}"


def reset_speak_cache() -> None:
    """清空「上次播报了什么」的记录，不删除音频文件。"""
    global _last_spoken
    with _lock:
        _last_spoken = None
    while True:
        try:
            _speak_queue.get_nowait()
        except queue.Empty:
            break


def request_speak(text: str, *, force: bool = False) -> str:
    zh_text = to_chinese(text)
    if not zh_text:
        return zh_text

    with _lock:
        if not force and zh_text == _last_spoken:
            return zh_text

    if not _cache_ready:
        init_speech()

    if not _audio_available:
        return zh_text

    _ensure_worker()
    _speak_queue.put((text, force))
    return zh_text


def speak(text: str, *, force: bool = False) -> str:
    zh = request_speak(text, force=force)
    wait_speech_done(timeout=15.0)
    return zh


def shutdown_speech() -> None:
    wait_speech_done(timeout=15.0)
    _speak_queue.put(None)
    if _worker is not None and _worker.is_alive():
        _worker.join(timeout=5.0)
    try:
        import pygame
        if pygame.mixer.get_init():
            pygame.mixer.quit()
    except Exception:
        pass


if __name__ == "__main__":
    print("=== 语音缓存测试 ===\n")
    t0 = time.time()
    init_speech()
    print(f"初始化耗时: {time.time() - t0:.1f}s\n")

    print("连续播报（应几乎无延迟）:")
    for label in ["hello", "thanks", "hospital", "danger", "help", "me"]:
        speak(label, force=True)

    print("\n重复「你好」应跳过（防重复）:")
    speak("hello")
    speak("hello")

    shutdown_speech()
    print("测试结束")
