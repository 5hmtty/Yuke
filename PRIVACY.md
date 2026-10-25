# 语键（Yuke）隐私政策（Privacy Policy）

最后更新：2026-10-27

语键（Yuke）是一款语音命令工具：**所有识别都在你的电脑上完成，音频与识别结果不上传**。

## 中文

**核心承诺：所有数据只存在你自己的电脑上，绝不上传。**

程序是便携式的：可写数据统一放在 **exe 同级的 `UserData\`** 目录。删除该目录（或整个程序目录）即删除全部数据；备份也只需复制这一个目录。

### 1. 麦克风音频

仅在你点击「开始识别」或「补录」时采集，只用于本机实时识别，**不上传、不传输、不发送到任何服务器**。

### 2. 录音（两类，请区分）

- **发音录音**（`UserData\command_recordings\`）：注册发音时自动保存，用于生成识别模板与日后重新生成。
  始终写入，使用 AES-256 加密；删除发音时同步删除对应录音。
- **会话录音**（`UserData\recordings\<时间>\audio.wav`）：主页「保存录音」开关控制，**默认关闭**。
  开启后以**明文 WAV** 保存当时采集的音频，仅用于你自行复盘；关闭开关不改变发音录音的保存。

### 3. 发音与配置

- `commands.json`、`groups.json`：发音名称、按键映射、触发方式、宏序列，**明文**保存，便于备份与手工编辑。
- `settings.json`：界面与引擎设置，**明文**保存，其中包括你选择的麦克风设备名、目标窗口标题关键字、窗口位置等。
- 识别模板（`embed_proto_v2.json`、`nuclei_templates_v2.json.enc`）：由你的录音生成的声音特征，**加密**保存。
  模板是派生数据，删除后会在下次启动时由发音录音自动重建。

### 4. 日志与崩溃记录（均在 `UserData\logs\`）

- 会话日志记录启动/停止、配置变更、触发事件、错误与诊断信息；其中可能包含你自定义的**发音名称**和**本机文件路径**。
- 崩溃记录包含异常码、程序版本、操作系统与架构；panic 记录额外包含源码位置与错误文本。**不含音频内容**。
- 日志仅保存在本机。向他人求助前请先自行检查内容。

### 5. 窗口标题

设置「目标窗口」时，程序会枚举当前可见的顶层窗口标题供你点选，并在触发按键前比对前台窗口标题。
标题只在内存中用于匹配；你选中的**标题关键字**会写入 `settings.json`。标题内容不会发送到任何地方。

### 6. 网络

程序不上传任何数据，也没有遥测、统计或广告。唯一联网行为是**「检查更新」**（可在设置中关闭）：
向 GitHub 查询最新版本号，只发送程序版本，不发送本机信息。查询会遵循系统代理设置。

### 7. 系统更改

- **全局热键**：向系统注册，用于暂停/恢复触发。
- **管理员权限**：见下方权限说明，不写入任何持久化设置，仅用于本次启动。

### 8. 卸载与备份

删除程序目录（含 `UserData\`）即删除全部数据。需要保留发音、录音或设置时，请在删除前复制 `UserData\`。

### 权限说明

- **麦克风**：语音识别必需；
- **管理员权限（UAC 弹窗）**：游戏常以管理员权限运行，Windows 会拦截普通程序的模拟按键——以管理员运行才能让按键进入游戏；
- **键盘模拟**：本软件的核心功能即「识别到发音后模拟按键」，使用 Windows 标准 `SendInput` 接口，不含任何内存读取或进程注入。

### 反作弊提醒

个别游戏的反作弊系统可能限制或拦截模拟键盘输入，使用前请确认目标游戏的用户协议；语键仅发送标准键盘事件，不读取内存、不注入任何进程。

**联系方式**：shmtty@qq.com

---

## English

Last updated: 2026-10-27

Yuke is a voice-command tool: **all recognition runs on your machine. Audio and recognition results are never uploaded.**

**Core promise: everything stays on your computer. Your data is never uploaded.**

The app is portable: all writable data lives in **`UserData\` next to the executable**. Delete that folder (or the program folder) to delete all data; copying that one folder is enough for backup.

### 1. Microphone audio

Captured only while recognition or phrase recording is active, processed entirely on-device — **never uploaded or transmitted**.

### 2. Recordings (two kinds)

- **Phrase recordings** (`UserData\command_recordings\`): saved automatically when you enroll a phrase; used to build recognition templates and to rebuild them later. Always written, AES-256 encrypted, deleted together with the phrase.
- **Session recordings** (`UserData\recordings\<time>\audio.wav`): controlled by the main-page "save recordings" toggle, **off by default**. When enabled, the captured audio is stored as a **plain WAV** for your own review. Turning it off does not affect phrase recordings.

### 3. Phrases and settings

- `commands.json`, `groups.json`: phrase names, key mappings, trigger modes, macros — **plain text** for easy backup and editing.
- `settings.json`: UI and engine settings, **plain text**, including the selected microphone name, the target-window title keyword and window position.
- Recognition templates (`embed_proto_v2.json`, `nuclei_templates_v2.json.enc`): voice features derived from your recordings, stored **encrypted**. They are derived data and are rebuilt from phrase recordings on the next launch if deleted.

### 4. Logs and crash records (under `UserData\logs\`)

- Session logs record start/stop, config changes, trigger events, errors and diagnostics. They may contain your **phrase names** and **local file paths**.
- Crash records contain the exception code, app version, OS and architecture; panic records additionally contain the source location and error text. **No audio content.**
- Logs stay on your machine. Please review them before sharing with anyone.

### 5. Window titles

When you set a "target window", the app enumerates visible top-level window titles for you to pick from, and compares the foreground window title before sending keys. Titles are used in memory for matching only; the **keyword you select** is written to `settings.json`. Titles are never sent anywhere.

### 6. Network

The app never uploads your data, and there is no telemetry, analytics or ads. The only network activity is the optional **"check for updates"** (can be disabled in Settings): it asks GitHub for the latest version number and sends only the app version, no machine information. The request follows your system proxy settings.

### 7. System changes (opt-in)

- **Global hotkey**: registered with the system to pause/resume triggering.
- **Administrator elevation**: see Permissions below; not persisted, applies to the current launch only.

### 8. Uninstall and backup

Deleting the program folder (including `UserData\`) deletes all data. Copy `UserData\` beforehand if you want to keep phrases, recordings or settings.

### Permissions

- **Microphone**: required for speech recognition;
- **Administrator (UAC prompt)**: many games run elevated and Windows blocks simulated input from non-elevated processes — running elevated lets keys reach the game;
- **Keyboard simulation**: the core function of this app, via the standard Windows `SendInput` API. No memory reading, no process injection.

### Anti-cheat notice

Some games' anti-cheat systems may restrict or block simulated keyboard input. Please review your game's terms of service. Yuke only sends standard keyboard events — it never reads memory or injects into any process.

**Contact**: shmtty@qq.com
