# 语键（Yuke）隐私政策（Privacy Policy）

最后更新：2026-09-22

语键（Yuke）是一款语音命令工具：**所有识别都在你的电脑上完成**。我们用最直白的方式说明数据处理方式：

## 中文

**核心承诺：所有数据只存在你自己的电脑上，绝不上传。**

1. **麦克风音频**：仅在你点击"开始识别"或"补录"时采集，只用于本机实时的发音识别，**不上传、不传输、不发送到任何服务器**。
2. **录音**：**默认关闭**（可在主页一键开启）。开启时，你注册发音时的录音会保存在本地数据目录中，并使用 AES-256 加密。删除发音时会同步删除对应录音。程序目录中不含你的任何数据。
3. **发音与配置**：你注册的发音、按键映射、触发方式、宏序列保存在本地（`commands.json`，明文可自行备份）；识别用的"指纹"（从你的录音生成的声音特征模板）同样只存本地并加密。
4. **崩溃日志**：程序异常退出时会在本地 `logs\` 目录写一份崩溃记录（不含任何录音内容），仅用于你反馈问题时自助排查。
5. **网络**：程序不会上传你的任何数据。**Microsoft Store 版**：更新通过 Microsoft Store 分发，应用自身不建立任何网络连接；**GitHub 直接下载版**：提供"检查更新"功能，仅在检查时向 GitHub 查询最新版本号，不发送任何本机信息，可在设置中关闭。除此之外无遥测、无统计、无广告。
6. **卸载与备份（分渠道）**：**Microsoft Store 版**——数据保存在系统的应用数据文件夹（`AppData\Local\Packages` 下语键的数据目录），**升级不删除数据，卸载会删除全部数据**；**GitHub zip 版**——数据保存在程序文件夹内，**删除程序文件夹即删除数据**。需要保留录音与发音时，请在卸载或删除前复制对应数据目录备份。

**权限说明**：
- **麦克风**：语音识别必需；
- **管理员权限（UAC 弹窗）**：游戏常以管理员权限运行，Windows 会拦截普通程序的模拟按键——以管理员运行才能让按键进入游戏；
- **键盘模拟**：本软件的核心功能即"识别到发音后模拟按键"，使用 Windows 标准 SendInput 接口，不含任何内存读取或注入。

**反作弊提醒**：个别游戏的反作弊系统可能限制或拦截模拟键盘输入，使用前请确认目标游戏的用户协议；语键仅发送标准键盘事件，不读取内存、不注入任何进程。

**联系方式**：shmtty@qq.com

---

## English

Last updated: 2026-09-22

Yuke is a voice-command tool: **all recognition runs on your machine**. In plain terms:

**Core promise: everything stays on your computer. Your data is never uploaded.**

1. **Microphone audio**: captured only while recognition or phrase recording is active, processed entirely on-device for local speech recognition — **never uploaded or transmitted**.
2. **Recordings**: **off by default** (one-click toggle on the main page to enable). When enabled, enrolled phrase recordings are stored in the local data folder with AES-256 encryption. Deleting a phrase deletes its recordings. The program folder itself contains none of your data.
3. **Phrases & settings**: your enrolled phrases, key mappings, trigger modes and macro sequences are stored locally (`commands.json`, plain text for easy backup); the voice "fingerprints" (templates generated from your recordings) are also stored locally, encrypted.
4. **Crash logs**: written locally to `logs\` on abnormal exit (contains no audio content), for your own troubleshooting.
5. **Network**: the app never uploads your data. **Microsoft Store version**: updates are delivered through Microsoft Store, and the app itself makes no network connections. **GitHub download version**: provides a "check for updates" feature that only queries GitHub for the latest version number — no information from your machine is sent, and it can be disabled in Settings. No telemetry, no analytics, no ads.
6. **Uninstall & backup (per channel)**: **Microsoft Store version** — data lives in the system per-app data folder (Yuke's folder under `AppData\Local\Packages`); **updates keep your data, uninstall deletes all of it**. **GitHub zip version** — data lives inside the program folder; **deleting the program folder deletes your data**. To keep recordings and phrases, copy the data folder before uninstalling or deleting.

**Permissions**:
- **Microphone**: required for speech recognition;
- **Administrator (UAC prompt)**: many games run elevated and Windows blocks simulated input from non-elevated processes — running elevated lets keys reach the game;
- **Keyboard simulation**: the core function of this app, via the standard Windows SendInput API. No memory reading, no process injection.

**Anti-cheat notice**: some games' anti-cheat systems may restrict or block simulated keyboard input. Please review your game's terms of service. Yuke only sends standard keyboard events — it never reads memory or injects into any process.

**Contact**: shmtty@qq.com