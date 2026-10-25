[English](README.md) | **简体中文**

# 语键 Yuke

Windows 上的离线语音命令工具。念一个短词，发送一次按键、一组按键或一段宏。面向需要额外输入方式的用户。

识别全程在本机完成，无网络请求；命令、录音与模板只存本地并加密。

## 特性

- **自定义录音**：每条命令录几次，就能触发
- **单字触发**：以字级判别为准，支持多字命令与一条命令多个发音
- **连续触发**：连续念多个命令，不需要每次重新开始识别
- **多种按键行为**：点击、按住切换、宏序列、互斥组、分组、全局热键或语音开关

## 运行

构建需要 Rust stable（MSVC）；运行需要 Windows 10 / 11 与管理员权限。

```bat
cargo build --release
powershell -ExecutionPolicy Bypass -File packaging/package_dist.ps1
dist\Yuke-<版本>\yuke.exe
```

- 打包脚本生成 `yuke.pak`（标准/抗噪编码器；包含随仓分发的 Silero VAD）
  并组装发行目录；`onnxruntime.dll` 等运行时依赖取自 `packaging/redist/`。
- 测试须串行（会占用音频设备与临时目录）：`cargo test --all-targets -- --test-threads=1`。
- 开发调试也可以散装运行：`xcopy /E /I assets\model target\release\model` +
  `copy packaging\redist\*.dll target\release\`（pak 优先于散装 model）。
- 推送 `1.2.6` 形式的 tag 会触发 CI 构建并发布 Release。

## 第一次使用

1. 设置 → 音频：选择麦克风。
2. 发音映射 → ＋新增命令：起名 → 补录 3 遍 → 重新生成发音模板 → 捕获按键。
3. 回主页点「开始识别」，念出该发音即触发。

## 文档

- 架构、数据流、命令行诊断：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 数据处理说明：[PRIVACY.md](PRIVACY.md)

## 许可

MIT，详见 [LICENSE](LICENSE)。模型与素材来源见 [ATTRIBUTIONS.md](ATTRIBUTIONS.md)。
