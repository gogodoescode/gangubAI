# GangubAI

A small voice-controlled study robot that runs on a Raspberry Pi 5.

Say **"Hi Gungu Bai"** and it:

- answers questions about your course slides (RAG over `ai/chatbot/Unit1/`, with a citation)
- starts a **timer** or **pomodoro** shown on its face screen
- **moves** when asked ("go forward", "turn left", "do a spin", "stop")
- **wanders** around on its own, using a cliff sensor so it does not fall off the table

It shows how it feels on an animated Pygame face. The same robot runs on real hardware
(L298N motors on Pi GPIO) or in Gazebo simulation.

## How it fits together

```
mic ─▶ wake word ─▶ record ─▶ whisper.cpp ─┬─▶ movement keywords ──▶ ai/robot.py ─▶ ROS 2
                                           └─▶ LangGraph + Groq ──┬─▶ retrieve_context (Chroma)
                                                                  ├─▶ move_robot / set_wander_mode ─▶ ai/robot.py
                                                                  └─▶ timer / pomodoro ─▶ face (UDP)
reply ─▶ Piper TTS ─▶ speaker          emotion ─▶ face (UDP localhost:8765)
```

| Path | What it does |
|---|---|
| `ai/voice/app.py` | Main voice loop |
| `ai/voice/wake_word.py`, `recorder.py`, `transcriber.py`, `tts.py` | OpenWakeWord, silence-based recording, whisper.cpp, Piper |
| `ai/voice/config.py` | Paths to models/binaries, audio settings, `TTS_USE_APLAY` |
| `ai/chatbot/graph.py` | LangGraph agent (Groq `gpt-oss-20b`, temperature 0.2) |
| `ai/chatbot/config.py` | Model, RAG folder, system prompt |
| `ai/chatbot/document_processor.py` | Indexes every PPTX/PDF in `ai/chatbot/Unit1/` into Chroma |
| `ai/chatbot/tools/` | `retrieve_context`, `move_robot`, `set_wander_mode`, `timer`, `pomodoro` |
| `ai/robot.py` | Sends `/motor_command` and `/wander_mode` to ROS 2 |
| `ai/frontend/` | Pygame face + timer/pomodoro overlays, listens on UDP |
| `ai/launcher/run_face_backend.py` | Starts face + voice loop with one command |
| `ros2_ws/src/gangubai_control/` | Motor controller (GPIO or sim) and wander controller |
| `ros2_ws/src/gangubai_description/` | URDF and Gazebo cliff test world |

## Setup

Requirements: Python 3.10+, ROS 2 Humble, a `GROQ_API_KEY` in `.env`.

```bash
pip install -r requirements.txt
```

These local assets are not in git. Set their paths in `ai/voice/config.py`:

- `whisper.cpp/` — built `whisper-cli` and `ggml-base.en.bin`
- `piper/` — Piper binary and voice `.onnx`
- `wakeword_models/` — `hi_gungu_bai.onnx` (+ optional `melspectrogram.onnx`, `embedding_model.onnx`)

Speech output: `TTS_USE_APLAY = True` for the Pi with the MAX98357A amp, `False` for laptop speakers.

The RAG index is built on first run into `.gangubai_db_hf/`.
Delete that folder after adding or removing slides so it is rebuilt.

## Run

Terminal 1 — ROS (pick one):

```bash
cd ros2_ws && source /opt/ros/humble/setup.bash && colcon build && source install/setup.bash

# Gazebo simulation
ros2 launch gangubai_control motor_control.launch.py simulate:=true wander_require_cliff_data:=false

# Real robot (keep cliff safety on)
ros2 launch gangubai_control motor_control.launch.py
```

Terminal 2 — face + voice:

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
python3 -m ai.launcher.run_face_backend --fullscreen
```

Voice logs go to `voice_backend.log`. Face keys: `1`–`8` emotions, `F` fullscreen, `Esc`/`Q` quit.

You can also run the parts separately: `python3 -m ai.voice.app` and `python3 -m ai.frontend.app`.
`langgraph.json` lets you open the agent in LangGraph Studio.
