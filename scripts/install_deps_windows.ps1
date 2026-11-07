[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu",
    [string]$WslInstallRoot = "~/.local/share/slicerdsurfer",
    [string]$WindowsInstallRoot = (Join-Path $env:LOCALAPPDATA "SlicerDSpace\wsl-runtime"),
    [string]$SlicerExecutable,
    [string]$FreeSurferLicensePath,
    [string]$ModelAssetsDir,
    [switch]$PlanOnly,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [switch]$AcceptFslLicense,
    [switch]$SkipLargeDownloads,
    [switch]$SkipFreeSurfer,
    [switch]$SkipModelAssets,
    [switch]$SkipApt,
    [switch]$SkipSlicerBridge,
    [switch]$SkipSlicerPythonInstall,
    [switch]$SkipUserConfigUpdate,
    [string]$PlanOutput,
    [string]$OrgRegUrl = "https://zenodo.org/record/2648292/files/ORG-RegAtlas-100HCP.zip?download=1",
    [string]$OrgRegSha256 = "c2b024980999b925ea896c49183995bfc97aeed8cbb648946dcbfa9e3ea6a5f1",
    [string]$Org800Url = "https://zenodo.org/record/2648292/files/ORG-800FC-100HCP.zip?download=1",
    [string]$Org800Sha256 = "8ab81aed998ff417f91284a33a3c9233afaaeb314caa32d520f6123c2fd9e741",
    [string]$TractCloudTrainUrl = "https://github.com/SlicerDMRI/TractCloud/releases/download/v1.0.0/TrainData_800clu800ol.tar.gz",
    [string]$TractCloudTrainSha256 = "72d70d4cf7a465a824ff421f7df29ef9116ab2c890ad428d312f020704cdfac7",
    [string]$TractCloudModelUrl = "https://github.com/SlicerDMRI/TractCloud/releases/download/v1.0.0/TrainedModel.tar.gz",
    [string]$TractCloudModelSha256 = "973911c073349f8b121ef67a4d378af17e7f7252883bed37588cec8ffa7c3a7d",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DependencyInstallerArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ScriptBoundParameters = @{} + $PSBoundParameters

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$WslScriptDir = Join-Path $ScriptDir "wsl"
$RuntimeInstaller = Join-Path $WslScriptDir "install_runtime.sh"
$BridgeSource = Join-Path $WslScriptDir "WslCommandBridge.cs"

$MiniforgeVersion = "26.3.2-3"
$MiniforgeSha256 = "848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c"
$MrtrixVersion = "3.0.8"
$FslEddyVersion = "2602.0"
$FslEddyBuild = "hb35a746_0"
$FslTopupVersion = "2203.5"
$FslTopupBuild = "ha905fa9_6"
$FreeSurferVersion = "7.4.1"
$TractQuerierCommit = "5c69c594d489859e08485110678dca7f6f4cba40"
$WmaCommit = "4532b911386d2d1bc7ebba63fdd13e65b1660623"
$TractCloudCommit = "94de6275d07d7b5eeb50e4cd4d6e85fde29d5fb0"
$AptPackages = @(
    "bash", "build-essential", "bzip2", "ca-certificates", "curl", "git",
    "libgl1", "libglib2.0-0", "libgomp1", "libsm6", "libxext6", "libxrender1",
    "python3", "tar", "tcsh", "unzip", "xvfb", "xz-utils"
)

function Assert-Sha256([string]$Name, [string]$Value) {
    if ($Value -notmatch '^[0-9a-fA-F]{64}$') {
        throw "$Name must be a 64-character SHA-256 digest."
    }
}

function Assert-ResourceOverride([string]$UrlName, [string]$ShaName) {
    if ($ScriptBoundParameters.ContainsKey($UrlName) -and -not $ScriptBoundParameters.ContainsKey($ShaName)) {
        throw "-$UrlName overrides a resource URL; provide -$ShaName for the exact replacement bytes."
    }
}

Assert-ResourceOverride "OrgRegUrl" "OrgRegSha256"
Assert-ResourceOverride "Org800Url" "Org800Sha256"
Assert-ResourceOverride "TractCloudTrainUrl" "TractCloudTrainSha256"
Assert-ResourceOverride "TractCloudModelUrl" "TractCloudModelSha256"
Assert-Sha256 "OrgRegSha256" $OrgRegSha256
Assert-Sha256 "Org800Sha256" $Org800Sha256
Assert-Sha256 "TractCloudTrainSha256" $TractCloudTrainSha256
Assert-Sha256 "TractCloudModelSha256" $TractCloudModelSha256

if ([string]::IsNullOrWhiteSpace($Distro) -or $Distro -match '[\r\n]') {
    throw "Distro must be a non-empty WSL distribution name."
}
if (-not (Test-Path -LiteralPath $RuntimeInstaller -PathType Leaf)) {
    throw "Missing WSL runtime installer: $RuntimeInstaller"
}
if (-not (Test-Path -LiteralPath $BridgeSource -PathType Leaf)) {
    throw "Missing WSL bridge source: $BridgeSource"
}

function Resolve-SlicerExecutable([string]$RequestedPath) {
    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        $candidates.Add($RequestedPath)
    }
    if (-not [string]::IsNullOrWhiteSpace($env:SLICER_EXECUTABLE)) {
        $candidates.Add($env:SLICER_EXECUTABLE)
    }
    $naMicRoot = Join-Path $env:LOCALAPPDATA "NA-MIC"
    if (Test-Path -LiteralPath $naMicRoot -PathType Container) {
        Get-ChildItem -LiteralPath $naMicRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            ForEach-Object { $candidates.Add((Join-Path $_.FullName "Slicer.exe")) }
    }
    $slicerOrgRoot = Join-Path $env:LOCALAPPDATA "slicer.org"
    if (Test-Path -LiteralPath $slicerOrgRoot -PathType Container) {
        Get-ChildItem -LiteralPath $slicerOrgRoot -Directory -Filter "Slicer *" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            ForEach-Object { $candidates.Add((Join-Path $_.FullName "Slicer.exe")) }
    }
    Get-ChildItem -LiteralPath $env:LOCALAPPDATA -Directory -Filter "Slicer *" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        ForEach-Object { $candidates.Add((Join-Path $_.FullName "Slicer.exe")) }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Resolve-PythonExecutable([string]$ResolvedSlicer) {
    if (-not [string]::IsNullOrWhiteSpace($env:PYTHON_BIN) -and
        (Test-Path -LiteralPath $env:PYTHON_BIN -PathType Leaf)) {
        return (Resolve-Path -LiteralPath $env:PYTHON_BIN).Path
    }
    if (-not [string]::IsNullOrWhiteSpace($ResolvedSlicer)) {
        $slicerDir = Split-Path -Parent $ResolvedSlicer
        foreach ($relative in @("PythonSlicer.exe", "bin\PythonSlicer.exe", "PythonSlicer")) {
            $candidate = Join-Path $slicerDir $relative
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return (Resolve-Path -LiteralPath $candidate).Path
            }
        }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    return $(if ($python) { $python.Source } else { $null })
}

$ResolvedSlicer = Resolve-SlicerExecutable $SlicerExecutable
$ResolvedPython = Resolve-PythonExecutable $ResolvedSlicer
$UserConfigPath = $(
    if (-not [string]::IsNullOrWhiteSpace($env:SLICERDSPACE_USER_CONFIG)) {
        [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($env:SLICERDSPACE_USER_CONFIG))
    } else {
        Join-Path $env:APPDATA "SlicerDSpace\user_config.json"
    }
)
$WindowsInstallRoot = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($WindowsInstallRoot))
if ([string]::IsNullOrWhiteSpace($ModelAssetsDir) -and (Test-Path -LiteralPath $UserConfigPath -PathType Leaf)) {
    try {
        $existingUserConfig = Get-Content -Raw -LiteralPath $UserConfigPath | ConvertFrom-Json
        $configuredModelAssets = $existingUserConfig.external_dependencies.model_assets_dir
        if (-not [string]::IsNullOrWhiteSpace([string]$configuredModelAssets)) {
            $ModelAssetsDir = [string]$configuredModelAssets
        }
    } catch {
        # The validated merge below reports malformed user JSON.  Do not use
        # an unparseable file merely to select an asset destination.
    }
}
if ([string]::IsNullOrWhiteSpace($ModelAssetsDir)) {
    $ModelAssetsDir = Join-Path $WindowsInstallRoot "model-assets"
}
$ModelAssetsDir = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($ModelAssetsDir))
$SkipResolvedModelAssets = [bool]($SkipModelAssets -or $SkipLargeDownloads)
if ([string]::IsNullOrWhiteSpace($PlanOutput)) {
    $PlanOutput = Join-Path $WindowsInstallRoot "install-plan.json"
}
$PlanOutput = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($PlanOutput))

$plan = [ordered]@{
    kind = "slicerdsurfer_windows_wsl_install_plan"
    version = 1
    execute = -not [bool]$PlanOnly
    distribution = $Distro
    wsl_install_root = $WslInstallRoot
    windows_install_root = $WindowsInstallRoot
    slicer_executable = $ResolvedSlicer
    python_executable = $ResolvedPython
    device = [ordered]@{
        requested = $Device
        resolved = "Deferred: GPU discovery runs inside WSL"
        torch_version = "2.11.0"
        cpu_index = "https://download.pytorch.org/whl/cpu"
        cuda_index = "https://download.pytorch.org/whl/cu126"
        fsl_cuda_package = "fsl-eddy-cuda=2602.0=h1a432fe_1"
    }
    fsl_license = [ordered]@{
        accepted = [bool]$AcceptFslLicense
        terms_url = "https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html"
    }
    new_user_brain_mask_backend = "pnl_cnn (existing user defaults are preserved)"
    apt = [ordered]@{
        execute_as = "wsl.exe --distribution <distro> --user root"
        skipped = [bool]$SkipApt
        packages = $AptPackages
    }
    pinned = [ordered]@{
        miniforge_version = $MiniforgeVersion
        miniforge_sha256 = $MiniforgeSha256
        mrtrix_version = $MrtrixVersion
        fsl_eddy_version = $FslEddyVersion
        fsl_eddy_build = $FslEddyBuild
        fsl_topup_version = $FslTopupVersion
        fsl_topup_build = $FslTopupBuild
        freesurfer_version = $FreeSurferVersion
        tract_querier_commit = $TractQuerierCommit
        wma_commit = $WmaCommit
        tractcloud_commit = $TractCloudCommit
    }
    resources = [ordered]@{
        org_reg = [ordered]@{ url = $OrgRegUrl; sha256 = $OrgRegSha256; skipped = [bool]$SkipLargeDownloads }
        org_800 = [ordered]@{ url = $Org800Url; sha256 = $Org800Sha256; skipped = [bool]$SkipLargeDownloads }
        tractcloud_train = [ordered]@{ url = $TractCloudTrainUrl; sha256 = $TractCloudTrainSha256; skipped = $false }
        tractcloud_model = [ordered]@{ url = $TractCloudModelUrl; sha256 = $TractCloudModelSha256; skipped = $false }
        model_assets = [ordered]@{
            profile = "core"
            path = $ModelAssetsDir
            skipped = $SkipResolvedModelAssets
        }
        freesurfer = [ordered]@{
            url = "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz"
            md5 = "4275a307a1a91043587f659389b3e9a8"
            skipped = [bool]($SkipLargeDownloads -or $SkipFreeSurfer)
            license_path = $(if ($FreeSurferLicensePath) { [IO.Path]::GetFullPath($FreeSurferLicensePath) } else { $null })
        }
    }
    bridge = [ordered]@{
        skipped = [bool]$SkipSlicerBridge
        source = $BridgeSource
        dependency_config = (Join-Path $WindowsInstallRoot "dependency-config.json")
        user_config = $UserConfigPath
        user_config_update_skipped = [bool]$SkipUserConfigUpdate
    }
}

$planDirectory = Split-Path -Parent $PlanOutput
[IO.Directory]::CreateDirectory($planDirectory) | Out-Null
[IO.File]::WriteAllText($PlanOutput, (($plan | ConvertTo-Json -Depth 8) + "`n"), (New-Object Text.UTF8Encoding($false)))
Write-Host "Install plan written: $PlanOutput"

if ($PlanOnly) {
    $plan | ConvertTo-Json -Depth 8
    exit 0
}
if (-not $AcceptFslLicense) {
    throw "Read the FSL license and pass -AcceptFslLicense before installation: https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html"
}

if (-not $SkipSlicerBridge -and [string]::IsNullOrWhiteSpace($ResolvedSlicer)) {
    throw "Slicer.exe was not found. Pass -SlicerExecutable, or use -SkipSlicerBridge explicitly."
}
if (-not $SkipSlicerPythonInstall -and [string]::IsNullOrWhiteSpace($ResolvedPython)) {
    throw "PythonSlicer.exe/python was not found. Set PYTHON_BIN or pass -SlicerExecutable."
}
if (-not $SkipFreeSurfer -and -not $SkipLargeDownloads) {
    if ([string]::IsNullOrWhiteSpace($FreeSurferLicensePath) -or
        -not (Test-Path -LiteralPath $FreeSurferLicensePath -PathType Leaf)) {
        throw "FreeSurfer 7.4.1 installation requires -FreeSurferLicensePath. Use -SkipLargeDownloads to defer it."
    }
}
if (-not [string]::IsNullOrWhiteSpace($FreeSurferLicensePath) -and
    -not (Test-Path -LiteralPath $FreeSurferLicensePath -PathType Leaf)) {
    throw "FreeSurfer license file not found: $FreeSurferLicensePath"
}

$WslCommand = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $WslCommand) {
    throw "wsl.exe is unavailable. Install WSL 2 and an Ubuntu distribution, then rerun this script."
}
$installedDistros = @(& wsl.exe --list --quiet) | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ }
if ($Distro -notin $installedDistros) {
    throw "WSL distribution '$Distro' is not installed. Installed: $($installedDistros -join ', ')"
}
$defaultUid = (& wsl.exe --distribution $Distro --exec id -u).Trim()
if ($LASTEXITCODE -ne 0 -or $defaultUid -eq "0") {
    throw "The default user for '$Distro' must be a non-root user. apt is invoked separately with --user root."
}

if (-not $SkipApt) {
    Write-Host "Installing Ubuntu packages as WSL root (no sudo)..."
    $aptCommand = "export DEBIAN_FRONTEND=noninteractive; apt-get update; apt-get install -y " + ($AptPackages -join " ")
    & wsl.exe --distribution $Distro --user root --exec bash -lc $aptCommand
    if ($LASTEXITCODE -ne 0) {
        throw "WSL apt dependency installation failed with exit code $LASTEXITCODE."
    }
}

function Convert-ToWslPath([string]$WindowsPath) {
    $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
    $portablePath = $resolved.Replace("\", "/")
    $converted = (& wsl.exe --distribution $Distro --exec wslpath -a -u $portablePath).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($converted)) {
        throw "Could not convert Windows path for WSL: $resolved"
    }
    return $converted
}

function Resolve-WslInstallRoot([string]$RequestedRoot) {
    if ($RequestedRoot.StartsWith("~/")) {
        $wslHome = (& wsl.exe --distribution $Distro --exec sh -c 'printf %s "$HOME"').Trim()
        if ($LASTEXITCODE -ne 0 -or -not $wslHome.StartsWith("/")) {
            throw "Could not resolve HOME for WSL distribution '$Distro'."
        }
        return ($wslHome.TrimEnd("/") + "/" + $RequestedRoot.Substring(2))
    }
    if (-not $RequestedRoot.StartsWith("/")) {
        throw "WslInstallRoot must be an absolute Linux path or begin with ~/ ."
    }
    return $RequestedRoot.TrimEnd("/")
}

$WslRepoRoot = Convert-ToWslPath $RepoRoot
$WslRuntimeInstaller = Convert-ToWslPath $RuntimeInstaller
$AbsoluteWslInstallRoot = Resolve-WslInstallRoot $WslInstallRoot
$WslLicense = ""
if (-not [string]::IsNullOrWhiteSpace($FreeSurferLicensePath)) {
    $WslLicense = Convert-ToWslPath $FreeSurferLicensePath
}

$runtimeArguments = @(
    "--repo-root", $WslRepoRoot,
    "--install-root", $AbsoluteWslInstallRoot,
    "--distro", $Distro,
    "--org-reg-url", $OrgRegUrl,
    "--org-reg-sha256", $OrgRegSha256,
    "--org-800-url", $Org800Url,
    "--org-800-sha256", $Org800Sha256,
    "--tractcloud-train-url", $TractCloudTrainUrl,
    "--tractcloud-train-sha256", $TractCloudTrainSha256,
    "--tractcloud-model-url", $TractCloudModelUrl,
    "--tractcloud-model-sha256", $TractCloudModelSha256
)
$runtimeArguments += @("--device", $Device, "--accept-fsl-license")
if ($WslLicense) { $runtimeArguments += @("--freesurfer-license", $WslLicense) }
if ($ResolvedSlicer) { $runtimeArguments += @("--slicer-windows-exe", $ResolvedSlicer) }
if ($SkipLargeDownloads) { $runtimeArguments += "--skip-large-downloads" }
if ($SkipFreeSurfer) { $runtimeArguments += "--skip-freesurfer" }

Write-Host "Installing pinned WSL runtime as the distribution's normal user..."
& wsl.exe --distribution $Distro --exec bash $WslRuntimeInstaller @runtimeArguments
if ($LASTEXITCODE -ne 0) {
    throw "WSL runtime installation failed with exit code $LASTEXITCODE."
}

$stateFile = "$AbsoluteWslInstallRoot/state/install-state.json"
$stateJson = (& wsl.exe --distribution $Distro --exec cat $stateFile) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "Could not read WSL install state: $stateFile"
}
$state = $stateJson | ConvertFrom-Json
if ($state.kind -ne "slicerdsurfer_windows_wsl_install_state") {
    throw "Unexpected WSL install state format."
}

if (-not $SkipResolvedModelAssets) {
    $modelDownloader = Join-Path $ScriptDir "download_model_assets.py"
    if (-not (Test-Path -LiteralPath $modelDownloader -PathType Leaf)) {
        throw "Missing model asset downloader: $modelDownloader"
    }
    [IO.Directory]::CreateDirectory($ModelAssetsDir) | Out-Null
    $wslModelDownloader = Convert-ToWslPath $modelDownloader
    $wslModelAssets = Convert-ToWslPath $ModelAssetsDir
    $modelLock = Join-Path $ModelAssetsDir "model-assets-lock.json"
    $wslModelLock = (& wsl.exe --distribution $Distro --exec wslpath -a -u ($modelLock.Replace("\", "/"))).Trim()
    Write-Host "Downloading/verifying the core model-asset profile through WSL..."
    & wsl.exe --distribution $Distro --exec "$($state.runtime_env)/bin/python" $wslModelDownloader `
        --asset-root $wslModelAssets --profile core --source url `
        --source-id slicerdsurfer_models_git --write-lockfile $wslModelLock
    if ($LASTEXITCODE -ne 0) {
        throw "Model asset installation failed with exit code $LASTEXITCODE."
    }
} else {
    Write-Host "Skipping the model-asset profile by request."
}

function Convert-LinuxPathToUnc([string]$LinuxPath) {
    if ([string]::IsNullOrWhiteSpace($LinuxPath)) { return $null }
    if (-not $LinuxPath.StartsWith("/")) { throw "Linux path is not absolute: $LinuxPath" }
    return "\\wsl.localhost\$Distro\" + $LinuxPath.TrimStart("/").Replace("/", "\")
}

function Build-BridgeBinary {
    $buildRoot = Join-Path $WindowsInstallRoot "bridge\_build"
    [IO.Directory]::CreateDirectory($buildRoot) | Out-Null
    $binary = Join-Path $buildRoot "slicerdsurfer-wsl-bridge.exe"
    $marker = "$binary.source-sha256"
    $sourceHash = (Get-FileHash -LiteralPath $BridgeSource -Algorithm SHA256).Hash.ToLowerInvariant()
    $mustBuild = -not (Test-Path -LiteralPath $binary -PathType Leaf)
    if (-not $mustBuild) {
        $recorded = $(if (Test-Path -LiteralPath $marker) { [IO.File]::ReadAllText($marker).Trim() } else { "" })
        $mustBuild = $recorded -ne $sourceHash
    }
    if ($mustBuild) {
        Remove-Item -LiteralPath $binary -Force -ErrorAction SilentlyContinue
        $sourceText = [IO.File]::ReadAllText($BridgeSource)
        Add-Type -TypeDefinition $sourceText -Language CSharp -OutputAssembly $binary -OutputType ConsoleApplication
        [IO.File]::WriteAllText($marker, "$sourceHash`n", (New-Object Text.UTF8Encoding($false)))
    }
    return $binary
}

function Install-BridgeExecutable(
    [string]$BridgeBinary,
    [string]$TargetPath,
    [string]$LinuxTarget,
    [string[]]$Mappings = @(),
    [string[]]$PrefixArguments = @(),
    [switch]$ProtectExisting
) {
    $parent = Split-Path -Parent $TargetPath
    [IO.Directory]::CreateDirectory($parent) | Out-Null
    $ownerMarker = "$TargetPath.slicerdsurfer-owner"
    if ($ProtectExisting -and (Test-Path -LiteralPath $TargetPath) -and -not (Test-Path -LiteralPath $ownerMarker)) {
        throw "Refusing to overwrite an existing executable not owned by SlicerDSpace: $TargetPath"
    }
    Copy-Item -LiteralPath $BridgeBinary -Destination $TargetPath -Force
    $configLines = @("distribution=$Distro", "target=$LinuxTarget")
    $configLines += "supervisor=$AbsoluteWslInstallRoot/bin/run-command.py"
    $configLines += "supervisor_python=$($state.runtime_env)/bin/python"
    foreach ($argument in $PrefixArguments) { $configLines += "arg=$argument" }
    foreach ($mapping in $Mappings) { $configLines += "map=$mapping" }
    [IO.File]::WriteAllLines("$TargetPath.conf", $configLines, (New-Object Text.UTF8Encoding($false)))
    [IO.File]::WriteAllText($ownerMarker, "SlicerDSpace WSL bridge v1`n", (New-Object Text.UTF8Encoding($false)))
}

$BridgeBinary = Build-BridgeBinary
$bridgeRoot = Join-Path $WindowsInstallRoot "bridge"
$mrtrixBridge = Join-Path $bridgeRoot "mrtrix"
$mrtrixCommands = @(
    "mrconvert", "dwi2response", "dwi2fod", "mtnormalise", "tckgen", "tcksift2",
    "tckstats", "tckconvert", "tcksample", "tck2connectome", "tckedit", "dwi2mask",
    "dwifslpreproc", "5ttgen", "5tt2gmwmi",
    "dwiextract", "mrcat", "python"
)
foreach ($command in $mrtrixCommands) {
    $linuxTarget = "$($state.runtime_env)/bin/$command"
    Install-BridgeExecutable $BridgeBinary (Join-Path $mrtrixBridge $command) $state.runtime_command_bridge -PrefixArguments @($linuxTarget)
    Install-BridgeExecutable $BridgeBinary (Join-Path $mrtrixBridge "$command.exe") $state.runtime_command_bridge -PrefixArguments @($linuxTarget)
}

$fslBridge = Join-Path $bridgeRoot "fsl"
foreach ($command in @("eddy", "eddy_cpu", "topup")) {
    $linuxTarget = "$($state.fsl.bin_dir)/$command"
    Install-BridgeExecutable $BridgeBinary (Join-Path $fslBridge $command) $state.runtime_command_bridge -PrefixArguments @($linuxTarget)
    Install-BridgeExecutable $BridgeBinary (Join-Path $fslBridge "$command.exe") $state.runtime_command_bridge -PrefixArguments @($linuxTarget)
}
if (($state.fsl.PSObject.Properties.Name -contains "cuda_ready") -and $state.fsl.cuda_ready) {
    foreach ($command in @("eddy_cuda", "eddy_cuda.exe")) {
        Install-BridgeExecutable $BridgeBinary (Join-Path $fslBridge $command) $state.runtime_command_bridge `
            -PrefixArguments @("$($state.fsl.bin_dir)/eddy_cuda")
    }
}

$tractCloudPythonBridge = Join-Path $bridgeRoot "tractcloud\python.exe"
Install-BridgeExecutable $BridgeBinary $tractCloudPythonBridge $state.tractcloud.python

$tractQuerierBridgeRoot = Join-Path $bridgeRoot "tract_querier"
$tractQuerierExe = Join-Path $tractQuerierBridgeRoot "bin\tract_querier.exe"
Install-BridgeExecutable $BridgeBinary $tractQuerierExe "$($state.runtime_env)/bin/tract_querier"
$tqScripts = Join-Path $tractQuerierBridgeRoot "scripts"
$tqPackage = Join-Path $tractQuerierBridgeRoot "tract_querier"
$tqData = Join-Path $tqPackage "data"
[IO.Directory]::CreateDirectory($tqScripts) | Out-Null
[IO.Directory]::CreateDirectory($tqData) | Out-Null
$tqShim = @'
from pathlib import Path
import subprocess
import sys

bridge = Path(__file__).resolve().parents[1] / "bin" / "tract_querier.exe"
raise SystemExit(subprocess.call([str(bridge), *sys.argv[1:]]))
'@
[IO.File]::WriteAllText((Join-Path $tqScripts "cli_tract_querier.py"), $tqShim, (New-Object Text.UTF8Encoding($false)))
[IO.File]::WriteAllText((Join-Path $tqPackage "__init__.py"), "# Windows probe shim; execution is delegated to the pinned WSL runtime.`n", (New-Object Text.UTF8Encoding($false)))
$tqSourceData = Convert-LinuxPathToUnc "$($state.tract_querier.root)/tract_querier/data"
if (Test-Path -LiteralPath $tqSourceData -PathType Container) {
    Copy-Item -Path (Join-Path $tqSourceData "*.qry") -Destination $tqData -Force
}

$wmaBridgeRoot = Join-Path $bridgeRoot "wma"
$wmaMappings = @()
if ($ResolvedSlicer -and $state.slicer_bridge) {
    $wmaMappings += "$ResolvedSlicer|$($state.slicer_bridge)"
}
Install-BridgeExecutable $BridgeBinary (Join-Path $wmaBridgeRoot "bin\python.exe") `
    $state.runtime_command_bridge -Mappings $wmaMappings `
    -PrefixArguments @("$($state.runtime_env)/bin/python")
$wmaCommands = @(
    "wm_apply_ORG_atlas_to_subject.sh",
    "wm_append_clusters_to_anatomical_tracts.py",
    "wm_assess_cluster_location_by_hemisphere.py",
    "wm_cluster_from_atlas.py",
    "wm_cluster_remove_outliers.py",
    "wm_diffusion_measurements.py",
    "wm_harden_transform.py",
    "wm_register_to_atlas_new.py",
    "wm_separate_clusters_by_hemisphere.py"
)
foreach ($command in $wmaCommands) {
    $linuxTarget = "$($state.wma.root)/bin/$command"
    Install-BridgeExecutable `
        $BridgeBinary `
        (Join-Path $wmaBridgeRoot "bin\$command") `
        $state.runtime_command_bridge `
        -Mappings $wmaMappings `
        -PrefixArguments @($linuxTarget)
    Install-BridgeExecutable `
        $BridgeBinary `
        (Join-Path $wmaBridgeRoot "bin\$command.exe") `
        $state.runtime_command_bridge `
        -Mappings $wmaMappings `
        -PrefixArguments @($linuxTarget)
}
Install-BridgeExecutable $BridgeBinary (Join-Path $wmaBridgeRoot "bin\xvfb-run") "/usr/bin/xvfb-run"

if (-not $SkipSlicerBridge) {
    $slicerDirectory = Split-Path -Parent $ResolvedSlicer
    $bashMappings = @()
    foreach ($relativePython in @("PythonSlicer.exe", "bin\PythonSlicer.exe", "PythonSlicer")) {
        $candidate = Join-Path $slicerDirectory $relativePython
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $bashMappings += "$((Resolve-Path -LiteralPath $candidate).Path)|$($state.runtime_env)/bin/python"
        }
    }
    $bashMappings += "$ResolvedSlicer|$($state.slicer_bridge)"
    Install-BridgeExecutable $BridgeBinary (Join-Path $slicerDirectory "bash.exe") "/bin/bash" $bashMappings -ProtectExisting
}

$externalDependencies = [ordered]@{
    mrtrix_bin_dir = $mrtrixBridge
    preprocessing_mrtrix_bin_dir = $mrtrixBridge
    preprocessing_fsl_bin_dir = $fslBridge
    tract_querier_root = $tractQuerierBridgeRoot
    wma_root = $wmaBridgeRoot
    tractcloud_root = Convert-LinuxPathToUnc $state.tractcloud.root
    tractcloud_python_executable = $tractCloudPythonBridge
}
if (Test-Path -LiteralPath $ModelAssetsDir -PathType Container) {
    $externalDependencies["model_assets_dir"] = $ModelAssetsDir
}
if ($ResolvedSlicer) { $externalDependencies["slicer_executable"] = $ResolvedSlicer }
if ($ResolvedPython) { $externalDependencies["python_executable"] = $ResolvedPython }
if ($state.freesurfer.home) {
    $externalDependencies["postprocess_freesurfer_home"] = Convert-LinuxPathToUnc $state.freesurfer.home
}
if ($state.wma.atlas_dir) {
    $externalDependencies["wma_atlas_dir"] = Convert-LinuxPathToUnc $state.wma.atlas_dir
}
if ($state.tractcloud.data_dir) {
    $externalDependencies["tractcloud_data_dir"] = Convert-LinuxPathToUnc $state.tractcloud.data_dir
}

$dependencyConfigPath = Join-Path $WindowsInstallRoot "dependency-config.json"
$dependencyConfig = [ordered]@{
    kind = "slicerdsurfer_dependency_config"
    version = 1
    external_dependencies = $externalDependencies
    windows_wsl = [ordered]@{
        distribution = $Distro
        state_file = Convert-LinuxPathToUnc $stateFile
        install_root = Convert-LinuxPathToUnc $AbsoluteWslInstallRoot
    }
}
[IO.Directory]::CreateDirectory($WindowsInstallRoot) | Out-Null
[IO.File]::WriteAllText(
    $dependencyConfigPath,
    (($dependencyConfig | ConvertTo-Json -Depth 8) + "`n"),
    (New-Object Text.UTF8Encoding($false))
)
Write-Host "Dependency config written: $dependencyConfigPath"

if (-not $SkipUserConfigUpdate) {
    if (Test-Path -LiteralPath $UserConfigPath -PathType Leaf) {
        try {
            $userConfig = Get-Content -Raw -LiteralPath $UserConfigPath | ConvertFrom-Json
        } catch {
            throw "Existing SlicerDSpace user config is not valid JSON; refusing to overwrite it: $UserConfigPath"
        }
    } else {
        $userConfig = [pscustomobject][ordered]@{
            kind = "slicerdsurfer_user_config"
            version = 1
            updated_at = $null
            ui = [pscustomobject][ordered]@{ language = "en"; first_run_wizard_shown = $false }
            recent = [pscustomobject][ordered]@{ subject_id = ""; subject_root = ""; output_root = "" }
            run_defaults = [pscustomobject][ordered]@{
                run_name = "Test1"
                targets = @("microstructure", "tractography", "bundles", "connectome")
                brain_mask_backend = "pnl_cnn"
                tractography_backend = "csd_act"
                bundle_backend = "wmql"
                microstructure_stages = @("dti")
                connectome_backend = "python"
                select_streamlines = 1000000
                auto_install_python_dependencies = $true
            }
            external_dependencies = [pscustomobject]@{}
        }
    }
    if ($null -eq $userConfig -or $userConfig -is [Array] -or $userConfig -is [string] -or $userConfig -is [ValueType]) {
        throw "SlicerDSpace user config must contain a JSON object: $UserConfigPath"
    }
    if (($userConfig.PSObject.Properties.Name -contains "kind") -and
        $null -ne $userConfig.kind -and
        $userConfig.kind -ne "slicerdsurfer_user_config") {
        throw "Unsupported SlicerDSpace user config kind '$($userConfig.kind)'; refusing to overwrite $UserConfigPath"
    }
    $mergedExternal = [ordered]@{}
    if ($userConfig.PSObject.Properties.Name -contains "external_dependencies" -and
        $null -ne $userConfig.external_dependencies) {
        foreach ($property in $userConfig.external_dependencies.PSObject.Properties) {
            $mergedExternal[$property.Name] = $property.Value
        }
    }
    foreach ($entry in $externalDependencies.GetEnumerator()) {
        $mergedExternal[$entry.Key] = $entry.Value
    }
    $externalObject = [pscustomobject]$mergedExternal
    if ($userConfig.PSObject.Properties.Name -contains "external_dependencies") {
        $userConfig.external_dependencies = $externalObject
    } else {
        $userConfig | Add-Member -NotePropertyName "external_dependencies" -NotePropertyValue $externalObject
    }
    # The standalone dependency profile must see preserved settings such as an
    # existing verified model-assets root, not only the newly created WSL
    # bridges.  Keep both JSON profiles aligned before running the doctor.
    $dependencyConfig["external_dependencies"] = $externalObject
    [IO.File]::WriteAllText(
        $dependencyConfigPath,
        (($dependencyConfig | ConvertTo-Json -Depth 8) + "`n"),
        (New-Object Text.UTF8Encoding($false))
    )
    foreach ($identity in @{ kind = "slicerdsurfer_user_config"; version = 1; updated_at = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ") }.GetEnumerator()) {
        if ($userConfig.PSObject.Properties.Name -contains $identity.Key) {
            $userConfig.($identity.Key) = $identity.Value
        } else {
            $userConfig | Add-Member -NotePropertyName $identity.Key -NotePropertyValue $identity.Value
        }
    }
    $userConfigDirectory = Split-Path -Parent $UserConfigPath
    [IO.Directory]::CreateDirectory($userConfigDirectory) | Out-Null
    $userConfigTemporary = "$UserConfigPath.tmp"
    [IO.File]::WriteAllText(
        $userConfigTemporary,
        (($userConfig | ConvertTo-Json -Depth 20) + "`n"),
        (New-Object Text.UTF8Encoding($false))
    )
    Move-Item -LiteralPath $userConfigTemporary -Destination $UserConfigPath -Force
    Write-Host "Slicer user config updated: $UserConfigPath"
}

if (-not $SkipSlicerPythonInstall) {
    $targets = @(
        "validation", "preprocessing", "segmentation", "parcellation", "surface", "surface_postprocess",
        "microstructure", "microstructure_freewater", "microstructure_noddi", "tractography", "bundles",
        "connectome", "saf", "wma", "tractcloud"
    ) -join ","
    $doctorOutput = Join-Path $WindowsInstallRoot "environment-doctor"
    $installArguments = @(
        (Join-Path $ScriptDir "install_dependencies.py"),
        "--targets", $targets,
        "--platform", "Windows",
        "--python-executable", $ResolvedPython,
        "--dependency-config", $dependencyConfigPath,
        "--output-dir", $doctorOutput,
        "--tractography-backend", "csd_act",
        "--connectome-backend", "mrtrix"
    )
    if ($DependencyInstallerArgs) { $installArguments += $DependencyInstallerArgs }

    # Probe first so the WSL runtime can download wheels for the exact Slicer
    # Python ABI.  This avoids depending on Slicer's embedded OpenSSL for PyPI
    # TLS while keeping the final install offline and reproducible by manifest.
    & $ResolvedPython @installArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Initial Slicer/Python dependency probe failed with exit code $LASTEXITCODE."
    }
    $doctorJsonPath = Join-Path $doctorOutput "environment_doctor.json"
    if (-not (Test-Path -LiteralPath $doctorJsonPath -PathType Leaf)) {
        throw "Dependency probe did not write: $doctorJsonPath"
    }
    $doctor = Get-Content -Raw -LiteralPath $doctorJsonPath | ConvertFrom-Json
    $missingPython = @(
        $doctor.summary.missing_python_packages |
            Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }
    )
    if ($missingPython | Where-Object { ([string]$_).StartsWith("git+") }) {
        throw "The managed Windows installer cannot build source-only Python requirements: $($missingPython -join ', ')"
    }

    $verificationArguments = @($installArguments) + @("--strict-exit-code")
    if ($missingPython.Count -gt 0) {
        $pythonVersion = (& $ResolvedPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
        $pythonBits = (& $ResolvedPython -c "import struct; print(struct.calcsize('P') * 8)").Trim()
        if ($LASTEXITCODE -ne 0 -or $pythonVersion -notmatch '^\d+\.\d+$' -or $pythonBits -ne "64") {
            throw "Could not resolve a supported 64-bit Slicer Python ABI (version='$pythonVersion', bits='$pythonBits')."
        }
        $pythonAbi = "cp$($pythonVersion.Replace('.', ''))"
        $wheelhouse = Join-Path $WindowsInstallRoot "python-wheelhouse\$pythonAbi"
        [IO.Directory]::CreateDirectory($wheelhouse) | Out-Null
        $wslWheelhouse = Convert-ToWslPath $wheelhouse
        $wslPython = "$($state.runtime_env)/bin/python"
        # pip's cross-target resolver selects Windows wheels but evaluates some
        # environment markers against the Linux downloader.  Declare the known
        # Windows / Slicer-Python 3.9 marker dependencies explicitly so the
        # offline wheelhouse is complete.
        $windowsMarkerRequirements = @(
            "pywin32>=306",
            "importlib-metadata>=6.0"
        )
        $wheelhouseRequirements = @($missingPython) + $windowsMarkerRequirements
        $downloadArguments = @(
            "-m", "pip", "download",
            "--dest", $wslWheelhouse,
            "--only-binary=:all:",
            "--platform", "win_amd64",
            "--implementation", "cp",
            "--python-version", $pythonVersion,
            "--abi", $pythonAbi
        ) + $wheelhouseRequirements
        Write-Host "Downloading verified Windows wheels through the managed WSL runtime..."
        & wsl.exe --distribution $Distro --exec $wslPython @downloadArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Windows wheelhouse download failed with exit code $LASTEXITCODE."
        }
        $wheelFiles = @(Get-ChildItem -LiteralPath $wheelhouse -File -Filter "*.whl")
        if ($wheelFiles.Count -eq 0) {
            throw "No wheels were downloaded to $wheelhouse"
        }
        $wheelManifest = [ordered]@{
            kind = "slicerdsurfer_windows_python_wheelhouse"
            version = 1
            generated_at = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
            python_version = $pythonVersion
            python_abi = $pythonAbi
            requirements = $missingPython
            windows_marker_requirements = $windowsMarkerRequirements
            files = @(
                $wheelFiles | Sort-Object Name | ForEach-Object {
                    [ordered]@{
                        name = $_.Name
                        size = $_.Length
                        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                    }
                }
            )
        }
        [IO.File]::WriteAllText(
            (Join-Path $wheelhouse "wheelhouse-manifest.json"),
            (($wheelManifest | ConvertTo-Json -Depth 6) + "`n"),
            (New-Object Text.UTF8Encoding($false))
        )
        $offlineArguments = @($installArguments) + @(
            "--execute", "--strict-exit-code", "--pip-no-index", "--pip-find-links", $wheelhouse
        )
        & $ResolvedPython @offlineArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Offline Slicer/Python dependency installation or verification failed with exit code $LASTEXITCODE."
        }
    } else {
        & $ResolvedPython @verificationArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Slicer/Python dependency verification failed with exit code $LASTEXITCODE."
        }
    }
}

$summary = [ordered]@{
    status = "completed"
    distribution = $Distro
    wsl_state = Convert-LinuxPathToUnc $stateFile
    dependency_config = $dependencyConfigPath
    user_config = $(if ($SkipUserConfigUpdate) { $null } else { $UserConfigPath })
    mrtrix_bridge = $mrtrixBridge
    fsl_bridge = $fslBridge
    fsl_ready = [bool]$state.fsl.ready
    slicer_bridge_installed = -not [bool]$SkipSlicerBridge
    freesurfer_ready = [bool]$state.freesurfer.ready
    wma_atlas_ready = [bool]$state.wma.atlas_ready
    tractcloud_data_ready = [bool]$state.tractcloud.data_ready
}
$summary | ConvertTo-Json -Depth 5
