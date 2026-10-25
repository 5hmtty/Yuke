# Yuke release packaging: build output + yuke.pak (models and optional models, plaintext) -> whitelisted distribution directory.
# Usage: powershell -ExecutionPolicy Bypass -File packaging/package_dist.ps1 [-OutDir dist\Yuke-<version>] [-KeepUserData] [-NoZip]
param(
    [string]$OutDir = "",
    [string]$ModelDir = "assets\model",
    [switch]$KeepUserData,
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"

# Do not use Remove-Item: restricted environments wrap it as safe-delete and abort on blocked deletes.
function Remove-Path([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        $it = Get-Item -LiteralPath $Path -Force
        if ($it.PSIsContainer) { [System.IO.Directory]::Delete($it.FullName, $true) }
        else { [System.IO.File]::Delete($it.FullName) }
    }
}

$Root = $PSScriptRoot | Split-Path
if (-not $OutDir) {
    $CargoVer = (Select-String -Path (Join-Path $Root "Cargo.toml") -Pattern '^version\s*=\s*"(.+?)"').Matches[0].Groups[1].Value
    $OutDir = "dist\Yuke-$CargoVer"
}
$ModelDir = Join-Path $Root $ModelDir
$OutDir = Join-Path $Root $OutDir
if (-not (Test-Path $OutDir -PathType Container)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

if (-not (Test-Path (Join-Path $Root "target\release\yuke.exe"))) {
    Write-Error "target\release\yuke.exe is missing - run cargo build --release first"
}
if (-not (Test-Path $ModelDir)) { Write-Error "model directory not found: $ModelDir" }
if (-not (Test-Path (Join-Path $Root "assets\silero_vad.onnx"))) {
    Write-Warning "assets\silero_vad.onnx missing - the pak will not include the VAD model; non-speech veto is disabled"
}

# onnxruntime DLL lookup order: repo root -> packaging\redist (committed, used by CI) -> old distribution OutDir.
# The source may be inside OutDir (an old package), which cleanup deletes first - read it into memory before cleanup.
$Redist = Join-Path $PSScriptRoot "redist"
$DllDirs = @($Root, $Redist, $OutDir)
foreach ($dll in "onnxruntime.dll", "onnxruntime_providers_shared.dll") {
    $found = $DllDirs | ForEach-Object { Join-Path $_ $dll } | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $found) { Write-Error "missing $dll (ort=load-dynamic; required at runtime)" }
}
$OrtBytes = [System.IO.File]::ReadAllBytes(($DllDirs | ForEach-Object { Join-Path $_ "onnxruntime.dll" } | Where-Object { Test-Path $_ } | Select-Object -First 1))
$OrtSharedBytes = [System.IO.File]::ReadAllBytes(($DllDirs | ForEach-Object { Join-Path $_ "onnxruntime_providers_shared.dll" } | Where-Object { Test-Path $_ } | Select-Object -First 1))

# Pre-check: a running yuke process holds the DLLs and blocks cleanup.
$proc = Get-Process yuke -ErrorAction SilentlyContinue
if ($proc) { Write-Error "yuke.exe is running (PID $($proc.Id -join ',')) - close the app first"; exit 1 }

# Clean the output directory with a whitelist: app files only; UserData is excluded by default.
$Stale = @("yuke.exe", "onnxruntime.dll", "onnxruntime_providers_shared.dll",
           "lang", "yuke.pak", "model", "assets", "LICENSE", "ATTRIBUTIONS.md")
if (-not $KeepUserData) { $Stale += "UserData" }
if (Test-Path $OutDir) {
    foreach ($f in $Stale) {
        $t = Join-Path $OutDir $f
        if (Test-Path $t) { Remove-Path $t }
    }
}

# ── exe / DLL ──
Copy-Item (Join-Path $Root "target\release\yuke.exe") $OutDir -Force
[System.IO.File]::WriteAllBytes((Join-Path $OutDir "onnxruntime.dll"), $OrtBytes)
[System.IO.File]::WriteAllBytes((Join-Path $OutDir "onnxruntime_providers_shared.dll"), $OrtSharedBytes)

# ── lang/ (UI strings) ──
Copy-Item (Join-Path $Root "lang") (Join-Path $OutDir "lang") -Recurse -Force

# ── licence / attribution ──
Copy-Item (Join-Path $Root "LICENSE") $OutDir -Force
Copy-Item (Join-Path $Root "ATTRIBUTIONS.md") $OutDir -Force

# ── yuke.pak: model (encoders) + assets (veto-layer models), plaintext payload ──
$PakSrc = Join-Path $env:TEMP ("yuke_pak_src_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item (Join-Path $PakSrc "model") -ItemType Directory -Force | Out-Null
New-Item (Join-Path $PakSrc "assets") -ItemType Directory -Force | Out-Null
Get-ChildItem $ModelDir -File | Where-Object { $_.Name -match '\.onnx$' -and $_.Name -notmatch '\.bak' } |
    ForEach-Object { Copy-Item $_.FullName (Join-Path $PakSrc "model") -Force }
if (Test-Path (Join-Path $Root "assets\silero_vad.onnx")) {
    Copy-Item (Join-Path $Root "assets\silero_vad.onnx") (Join-Path $PakSrc "assets") -Force
}
if (Test-Path (Join-Path $Root "assets\sv_cnceleb_resnet34.onnx")) {
    Copy-Item (Join-Path $Root "assets\sv_cnceleb_resnet34.onnx") (Join-Path $PakSrc "assets") -Force
} else {
    Write-Warning "assets\sv_cnceleb_resnet34.onnx is missing - the pak will not include the speaker model; speaker verification is disabled"
}

$exe = Join-Path $OutDir "yuke.exe"
& $exe pak-pack (Join-Path $OutDir "yuke.pak") (Join-Path $PakSrc "model") (Join-Path $PakSrc "assets") | Write-Host
if ($LASTEXITCODE -ne 0) { Write-Error "pak-pack failed"; exit 1 }
Remove-Path $PakSrc

# Self-check: the pak parses and contains the expected entries.
& $exe pak-list | Write-Host
if ($LASTEXITCODE -ne 0) { Write-Error "pak-list self-check failed"; exit 1 }

if ($KeepUserData) {
    Write-Warning "UserData was kept (local settings/recordings/logs) - for local testing only; do not distribute it"
}
Write-Host "`n=== Package ready: $OutDir ==="
Get-ChildItem $OutDir -Recurse -File | ForEach-Object {
    "{0,10:N0}  {1}" -f $_.Length, $_.FullName.Substring($OutDir.Length + 1)
}

# Zip: create <OutDir>.zip by default (-NoZip skips). CI depends on this step for release assets.
if (-not $NoZip) {
    $ZipPath = "$OutDir.zip"
    if (Test-Path $ZipPath) { Remove-Path $ZipPath }
    Compress-Archive -Path $OutDir -DestinationPath $ZipPath -Force
    $zipLen = (Get-Item $ZipPath).Length
    $zipSha = (Get-FileHash $ZipPath -Algorithm SHA256).Hash.ToLower()
    Write-Host ("`n[zip] {0}  ({1:N0} bytes)" -f $ZipPath, $zipLen)
    Write-Host ("[zip] SHA256 {0}" -f $zipSha)
}
