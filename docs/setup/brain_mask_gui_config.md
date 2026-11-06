<div align="center">

# Brain Mask GUI and Configuration

[Project](../../README.md) | [Documentation](../README.md) | [Shared Configuration](USER_CONFIG.md) | [CLI Options](brain_mask_cli.md)

</div>

---

The selector is only in **Advanced settings > Preprocessing & mask > Brain
masking**, not in the main Inputs or Products sections.

| Display | Saved/CLI value |
| --- | --- |
| PNL CNN (dMRI), default | `pnl_cnn` |
| DIPY median-Otsu, model-free Python option | `dipy` |
| SynthStrip | `synthstrip` |
| MRtrix dwi2mask | `mrtrix` |
| Provided mask only | `provided` |

PNL CNN licence: All or portions of this licensed product (such portions are
the "Software") have been obtained under license from The Brigham and Women's
Hospital, Inc. and are subject to the following terms and conditions:
see [PNL CNN brain mask licence](../licenses/pnl_brain_mask.md) for the full
terms (also in `shared/pnl_cnn/LICENSE.txt`).

The selector controls generation of a missing mask. Supplied masks are
respected. A missing dependency or required mask is an error, not permission
to fall back to another backend. Selecting PNL CNN does not install or imply
the availability of its models or environment.

The adjacent **Mask device** selector defaults to `auto`: PNL CNN and SynthStrip
prefer the first CUDA GPU visible to their own external Python. Select `cuda`
to require GPU or `cpu` to explicitly disable it. The setting persists as
`run_defaults.brain_mask_device` and is forwarded for both single and batch runs.
GPU runtime errors are reported, not silently retried on CPU. MRtrix dwi2mask
is CPU-only; the selector does not invent a GPU implementation for it.
See [GPU execution policy](GPU_EXECUTION.md).

On Linux x86_64, **Environment and installation > Prepare selected
analysis environment** can install the pinned author-compatible PNL runtime,
weights and template after terms confirmation, then fill all four paths.
This is the official PNL Python masking backend, not a rewrite of PNLpipeline.
See [managed setup](MANAGED_SETUP.md) for repair, verification and platform limits.
For an existing environment, configure **External Runtime Paths**:

| Setting | Browse Mode |
| --- | --- |
| `pnl_mask_python` | Python executable file in the user's PNL environment |
| `pnl_mask_script` | Official `dwi_masking.py` file |
| `pnl_mask_model_dir` | Model directory |
| `pnl_mask_ants_bin_dir` | ANTs binary directory |
| `synthstrip_python` | Optional CUDA-compatible Python with PyTorch/surfa; defaults to FreeSurfer fspython |

SynthStrip reuses `postprocess_freesurfer_home`; MRtrix reuses `mrtrix_bin_dir`.
The MRML `brainMaskBackend` is saved as `run_defaults.brain_mask_backend` in
the shared user JSON and in Slicer user settings. The four external paths are
saved under `external_dependencies`, and dependency JSON import/export uses
the same keys. Single-subject and batch background commands explicitly pass
`--brain-mask-backend`; external path flags use hyphens instead of underscores.
Explicit CLI options override saved values.

The GUI environment check requires the selected generator when a known
canonical input is missing its mask or automatic masking is selected with
preprocessing. The GUI's default motion correction may require a fresh mask in
corrected space, even if a supplied input mask exists. With preprocessing off,
a supplied canonical mask does not require generator dependencies. `provided`
does not require an inference runtime for preprocessing; a known missing
canonical mask is still an error. These options select **brain masking only**,
not the backend for the entire pipeline.

---

## Verification

```bash
python -m unittest Testing.Python.test_brain_mask_gui_config
python Testing/Python/run_brain_mask_gui_smoke.py \
  --slicer /path/to/Slicer --xvfb /path/to/xvfb-run \
  --report /tmp/new-brain-mask-gui-smoke
```

The runtime smoke uses a new, isolated settings directory and private Slicer
revision settings, runs no inference, and starts Slicer twice. It checks the
default, Advanced placement, all four choices, node/UI synchronization,
debounced saves, EN/ZH labels, external-path persistence, and restoration in a
new process. Its report and screenshots are written under `--report`.

---

## 中文

脑掩膜选择器仅位于“高级设置 > 预处理与掩膜 > 脑掩膜方法”，默认显示
`PNL CNN (dMRI)`。可选 SynthStrip、MRtrix dwi2mask、无需权重或外部程序的 DIPY median-Otsu，或“仅使用提供的掩膜”。
已有的用户掩膜会被尊重；依赖或掩膜缺失时明确报错，不会自动换用其他后端。
选择默认方法不代表 PNL 模型或运行环境已安装。

旁边的“掩膜计算设备”默认 `auto`，PNL CNN 和 SynthStrip 优先使用各自外部
Python 中可见的第一个 CUDA GPU；`cuda` 要求 GPU 可用，`cpu` 显式使用 CPU。
选择会持久化并传给单被试、批处理和预处理流程。GPU 报错不会静默改用 CPU。
MRtrix dwi2mask 本身仅支持 CPU。详见[GPU 策略](GPU_EXECUTION.zh-CN.md)。

Linux x86_64 可点击“环境与安装 > 准备所选分析环境”，确认条款后自动准备
固定版本的官方 PNL Python 环境、权重与模板，校验后填写路径；并非重写 PNLpipeline。
修复、验证和平台限制见[自动配置](MANAGED_SETUP.zh-CN.md)。
已有环境仍可在“外部运行路径”配置用户自己的 PNL Python
可执行文件、官方 `dwi_masking.py`、模型目录及 ANTs bin 目录。SynthStrip
共用 FreeSurfer 主目录。选择和路径会写入共享用户 JSON，供 GUI 与 CLI
复用；上述运行时测试隔离用户设置，并通过两次启动检查持久化。
