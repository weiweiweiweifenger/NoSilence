# NoSilence · 无声之声

> **让每一帧手语自然、流畅地转化为文字与语音**  
> 为听障人士搭建一座实时沟通的桥梁，支持 6 类常用手势识别，并借助大模型将连续手势润色为自然语句。

---

## 📌 项目背景

中国约有 2780 万听障人士，手语是他们的第一语言，但社会大众中手语普及率不足 1%。在就医、求助、日常交流等场景中，他们常因无法被“听懂”而孤立无援。**NoSilence** 通过计算机视觉与轻量级深度学习，在普通笔记本上即可实现实时手势识别与语音播报，并结合 AI 润色提升表达连贯性，让“无声”拥有“有声”的力量。

---

## ✨ 功能特性

- **实时手势识别** – 支持 6 类词汇（你好、谢谢、帮助、医院、危险、我），单帧推理延迟 < 100ms
- **双进程分离架构** – OpenCV 负责视频处理与推理，Streamlit 独立展示结果，互不阻塞
- **稳定帧过滤** – 5 帧一致且置信度 ≥ 0.45 才触发输出，有效抑制抖动
- **语音播报** – 使用 edge-tts 生成缓存，pygame 内存播放，延迟 < 200ms，且同一手势不重复播报
- **Web 仪表盘** – 实时显示当前识别结果、历史记录（最近 10 条）、统计信息，无需传输视频流
- **AI 润色** – 将连续识别的手势词汇序列（去重、按时间排序）通过 Qwen / OpenAI 整理为通顺中文，无 API 时自动降级为规则演示
- **数据采集与训练一体化** – 内置采集脚本，支持单手/双手混合数据，一键训练 MLP 模型

---

## 🧱 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                 双进程分离 · 本地 JSON 文件通信               │
├─────────────────────────────┬───────────────────────────────┤
│     进程 A (OpenCV)          │      进程 B (Streamlit)        │
│  gesture_recognize.py        │  gesture_web.py               │
│  - 摄像头采集                 │  - 读取 runtime_state.json     │
│  - MediaPipe 手部检测         │  - 展示当前结果 / 历史 / 统计   │
│  - 特征提取 (127维)           │  - 提供 AI 润色按钮            │
│  - MLP 推理                   │  - 发送命令 (clear_history)    │
│  - 稳定帧过滤                 │                              │
│  - 语音播报                   │                              │
│  - 写入 runtime_state.json   │                              │
│  - 读取 runtime_command.json │                              │
└─────────────────────────────┴───────────────────────────────┘
                        ↕
              data/runtime_state.json
              data/runtime_command.json
```

> **设计核心**：视频处理与界面完全隔离，通过简单可靠的文件 IPC 耦合，保证实时性与可维护性。

---

## 🚀 快速开始

### 环境要求
- Python 3.9+
- Windows 10/11（或 Linux，但需调整摄像头索引与字体路径）
- USB 摄像头（≥30 万像素）

### 安装依赖
```bash
git clone https://github.com/yourname/nosilence.git
cd nosilence
pip install -r requirements.txt
```

### 1. 采集手势数据（可选，也可使用预训练模型）
```bash
python gesture_collect.py
```
- 按数字键 `1`~`6` 选择手势类别
- 按 `空格键` 开始录制（倒计时 3s → 稳定提示 0.5s → 录制 10s）
- 数据自动保存至 `data/gesture_data_2hands.csv`

### 2. 训练模型
```bash
python train_gesture.py
```
训练完成后生成 `gesture_model.pth`（包含权重、类别列表、标准化参数）。

### 3. 启动识别进程（OpenCV 窗口）
```bash
python gesture_recognize.py
```
- 摄像头打开，实时显示手部骨架与识别结果
- 按 `q` 退出

### 4. 启动 Web 面板（另一个终端）
```bash
streamlit run gesture_web.py
```
浏览器访问 `http://localhost:8501` 即可查看仪表盘。

---

## 📁 项目结构

```
NoSilence/
├── gesture_collect.py          # 数据采集
├── gesture_features.py         # 特征提取（训练/推理共用）
├── train_gesture.py            # 模型训练
├── gesture_recognize.py        # 实时识别主程序
├── gesture_speech.py           # 语音播报
├── gesture_state.py            # IPC 状态读写
├── gesture_web.py              # Streamlit UI
├── gesture_polish.py           # AI 润色
├── hand_detection.py           # 调试用：单独检测手部
├── reset_dataset.py            # 重置数据集
├── data/
│   ├── gesture_data_2hands.csv # 采集数据
│   ├── runtime_state.json      # 运行时状态（IPC）
│   ├── runtime_command.json    # 控制命令（IPC）
│   └── tts_cache/              # 语音缓存 MP3
├── gesture_model.pth           # 训练好的模型
└── hand_landmarker.task        # MediaPipe 模型文件
```

---

## ⚙️ 技术栈

| 层次           | 技术                        | 理由                                   |
|----------------|-----------------------------|----------------------------------------|
| 视频采集       | OpenCV 4.x                  | 跨平台，低延迟摄像头读取                 |
| 手部检测       | MediaPipe Hand Landmarker   | CPU 实时，21 关键点，无需 GPU           |
| 分类模型       | PyTorch MLP                 | 轻量推理，易于保存/加载                  |
| 语音合成       | edge-tts                    | 免费，自然音质，缓存后离线播放            |
| 语音播放       | pygame.mixer                | 内存加载，延迟 < 50ms                   |
| Web 框架       | Streamlit                   | 快速构建数据面板，无需前端                |
| AI 润色        | DashScope Qwen / OpenAI     | 中文能力强，降级演示规则保证离线可用      |
| 进程通信       | JSON 文件 + 重试写入        | 简单可靠，避免 socket 复杂性             |

---

## 📊 关键配置参数

| 参数                     | 值     | 说明                         |
|--------------------------|--------|------------------------------|
| `STABLE_FRAMES`          | 5      | 稳定所需连续帧数              |
| `MIN_SPEAK_CONFIDENCE`   | 0.45   | 播报置信度阈值                |
| `FLUSH_INTERVAL`         | 0.15s  | 状态文件写入间隔              |
| `WEB_REFRESH_INTERVAL`   | 0.4s   | Web 面板刷新间隔              |
| `EPOCHS`                 | 100    | 训练轮数                     |
| `BATCH_SIZE`             | 32     | 批大小                       |
| `LEARNING_RATE`          | 1e-3   | 学习率                       |
| `HISTORY_MAX_LEN`        | 20     | 历史记录最大条数              |

---

## 🧪 演示与润色

- **AI 润色**：在 Web 面板点击 “AI润色” 按钮，系统会从历史记录中提取去重词汇（按时间正序），优先调用 Qwen API，无 Key 则使用内置规则（如 `["我","医院","帮助"]` → “我需要去医院，希望得到帮助。”）。
- **环境变量配置**（可选）：
  - `DASHSCOPE_API_KEY`（阿里云 Qwen）
  - `OPENAI_API_KEY`（OpenAI 备选）

---

## ⚠️ 注意事项

1. 首次运行语音播报需联网（edge-tts 下载音频），之后完全离线。
2. 确保摄像头光照充足，手部完整出现在画面内。
3. 若模型文件不存在，请先运行 `train_gesture.py`。
4. 双进程需在同一机器上运行，通过本地 JSON 通信，无需网络。
5. 手势限定为 6 类静态/简单动态手势，不覆盖全部手语词汇。

---


## 🙏 致谢

本项目基于 MediaPipe、PyTorch、Streamlit 等优秀开源工具构建，旨在为信息无障碍事业贡献一份微薄之力。欢迎提交 Issue 和 PR，共同完善。

---

