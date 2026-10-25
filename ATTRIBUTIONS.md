# ATTRIBUTIONS / 版权与来源声明

Yuke（语键）的运行时依赖与部分素材来自第三方项目及公开语料。
本文件按类别声明来源与许可状态；**加粗【】内为需要注意的许可条款**。

> This file attributes the third-party material and open-source
> dependencies used by Yuke. Items marked 【】 are license notes.

---

## 1. 识别模型 / Recognition model

- 编码器（`assets/model/encoder-v2*.onnx`，标准版与抗噪版）：**本项目基于下述基座模型微调**，
  与源码一起在本仓库分发。
- 基座模型（Base model）：[ModelScope `pkufool/icefall-kws-zipformer-zh-en-3M-2025-12-20`](https://modelscope.cn/models/pkufool/icefall-kws-zipformer-zh-en-3M-2025-12-20)
  （icefall / sherpa-onnx 生态的 zipformer 关键词 spotting 模型，Apache-2.0）。
  再分发时保留本节归属。
- The encoders (`assets/model/encoder-v2*.onnx`, standard and noise-robust variants)
  are fine-tuned from the Apache-2.0 icefall/sherpa-onnx zipformer KWS model above
  and are distributed with this repository. Keep this attribution when redistributing.

## 2. 数据来源 / Data sources

- 部分微调数据源自 **WenetSpeech**（CC BY 4.0）。
- Part of the fine-tuning data is derived from **WenetSpeech** (CC BY 4.0).

## 3. 触发否决层模型 / Veto-layer models（Silero 随仓库分发；CN-Celeb 可选）

- **Silero VAD**（`assets/silero_vad.onnx`）：人声活动检测模型，用于拦截非人声
  （键盘声、呼吸声等）误触发。随仓库分发，打包装入 yuke.pak。
  来源：[snakers4/silero-vad](https://github.com/snakers4/silero-vad)，许可 MIT。
  Silero VAD (`assets/silero_vad.onnx`): voice activity detection used to reject
  non-speech triggers. Bundled with this repository. Source: snakers4/silero-vad, MIT licensed.

- **wespeaker CN-Celeb ResNet34**（`assets/sv_cnceleb_resnet34.onnx`）：说话人嵌入
  模型，用于"声纹门"（实验功能，默认关闭——触发候选窗与本机注册质心比对，音频不出本机）。
  可选，不随仓库分发；缺失时该功能自动停用。
  来源：[wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker)（Apache-2.0）
  发布的 CN-Celeb 预训练权重。
  **【训练数据 CN-Celeb 官方条款限非商业的教育/科研/技术开发用途；本项目为免费
  分发的无障碍工具、非商业使用，模型仅在本机推理。如将本软件用于商业用途，
  请自行复核该数据许可。】**
  wespeaker CN-Celeb ResNet34 (`assets/sv_cnceleb_resnet34.onnx`): speaker
  embedding model for the voice-print gate (experimental, off by default —
  comparison against the on-device enrolled centroid; audio never leaves the
  machine). Source: pretrained weights published by wenet-e2e/wespeaker
  (Apache-2.0). 【The CN-Celeb training data terms are non-commercial
  (education/research/development); this project is a free accessibility tool
  with non-commercial use, inference on-device only. Re-verify the data license
  if you use this software commercially.】

## 4. 运行时依赖 / Runtime dependencies

| 依赖 (Dependency) | 版本 | 许可 (License) |
| --- | --- | --- |
| onnxruntime（外部 DLL，微软） | 1.24 | MIT |
| ort（onnxruntime Rust 绑定） | 2.0.0-rc.13 | MIT OR Apache-2.0 |
| eframe / egui（桌面 UI） | 0.31.1 | MIT OR Apache-2.0 |
| cpal（音频采集） | 0.15.3 | Apache-2.0 |
| global-hotkey（全局热键） | 0.6.4 | Apache-2.0 OR MIT |
| anyhow | 1.0.104 | MIT OR Apache-2.0 |
| ctrlc | 3.5.2 | MIT OR Apache-2.0 |
| chrono | 0.4.45 | MIT OR Apache-2.0 |
| hound（WAV 读写） | 3.5.1 | Apache-2.0 |
| ndarray | 0.17.2 | MIT OR Apache-2.0 |
| rustfft | 6.4.1 | MIT OR Apache-2.0 |
| rtrb（实时无锁环形队列） | 0.3.4 | MIT OR Apache-2.0 |
| serde / serde_json | 1.0.229 / 1.0.151 | MIT OR Apache-2.0 |
| windows（Win32 绑定） | 0.58.0 | MIT OR Apache-2.0 |
| windows-core（windows 绑定） | 0.58.0 | MIT OR Apache-2.0 |
| egui_extras（egui 扩展控件） | 0.31.1 | MIT OR Apache-2.0 |
| sonora / sonora-common-audio（WebRTC APM 的 Rust 移植，回声消除） | 0.2.0 | BSD-3-Clause |
| ureq（检查更新 HTTP 客户端） | 2.8.0 | MIT OR Apache-2.0 |
| winresource（构建期图标/版本资源） | 0.1.31 | MIT |

以上许可文本的完整副本随各上游项目分发；crates.io 依赖的源码可经
`cargo vendor` 或 crates.io 获取。完整依赖树（含传递依赖）见 `Cargo.lock`。

Full license texts are available from each upstream project; the complete
dependency tree (including transitive dependencies) is recorded in `Cargo.lock`.

## 5. 本项目许可 / This project's license

Yuke 本体以 MIT 许可开源，见 `Cargo.toml` 的 `license = "MIT"`。
Yuke itself is open-sourced under the MIT license (see `license = "MIT"` in `Cargo.toml`).
