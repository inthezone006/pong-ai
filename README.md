# 🏓 Pong AI
A classic Pong game with a TensorFlow-powered AI opponent, built in C++ for Win32 using Visual Studio 2022. No Python required at runtime—just unzip and play!

## 🎯 Project Goal

Create a lightweight Windows executable demonstrating how to embed a TensorFlow model in a native Win32 C++ game loop.

## ✨ Features

- 🎮 **Native Win32 Game**: Smooth 60 FPS Pong rendered with GDI.

- 🤖 **TensorFlow AI**: Paddle movement decided by a tiny neural network via the TensorFlow C API.

- 🐍 **Python Training Script**: `generate_pong_data.py` builds & trains the model in a virtual environment.

- 📦 **Portable Distribution**: Ship `PongWin32AI.exe`, `tensorflow.dll` & `saved_pong_ai/`; runs on Windows 10/11 x64.

## 🚀 Getting Started

### 🧰 Prerequisites

- 💻 Windows 10/11 x64
- 🛠️ Visual Studio 2022 (Desktop C++ workload)
- 🐍 Python 3.8+
- 🗄️ TensorFlow C API
- 📦 Visual C++ Redistributable 2022

### 🛠️ Installation

1. 📥 Clone the repo:
    ```bash
    git clone https://github.com/inthezone006/pong-ai
    cd pong-ai
    ```

2. 🐍 (Optional) Create & activate a venv:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

3. 📈 Install **TensorFlow** & train:
    ```bash
    pip install --upgrade pip
    pip install tensorflow
    python generate_pong_data.py
    ```

4. 🗂️ Download & extract the **TensorFlow C API (CPU)** from https://www.tensorflow.org/install/lang_c#download_the_tensorflow_c_library into `tf_c_api/` so you have `include/`, `lib/` (with `.lib`), and `bin/` (with `tensorflow.dll`).

5. 🔧 Open `PongWin32AI\PongWin32AI.sln` in Visual Studio.

    - C/C++ → General → Additional Include Directories → `..\tf_c_api\include`

    - Linker → General → Additional Library Directories → `..\tf_c_api\lib`

    - Linker → Input → Additional Dependencies → `tensorflow.lib`

    - Debugging → Working Directory → `$(OutDir)`

6. 📋 Copy

    - `saved_pong_ai/` → `PongWin32AI\x64\Debug\saved_pong_ai\`

    - `tf_c_api\bin\tensorflow.dll` → `PongWin32AI\x64\Debug\`

7. 🔨 Build & Run (F5) in x64-Debug.

8. 😄 Have fun!