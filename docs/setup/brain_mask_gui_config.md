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

The adjacent **Mask device** selector defaults to `auto`: PNL CNN
prefers the first CUDA GPU visible to its own external Python. Select `cuda`
to require GPU or `cpu` to explicitly disable it. The setting persists as
`run_defaults.brain_mask_device` and is forwarded for both single and batch runs.
GPU runtime errors are reported, not silently retried on CPU.
See [GPU execution policy](GPU_EXECUTION.md).

The advanced **PNL route** selector (shown only for PNL CNN) is `auto`, `managed`
or `native (experimental)`: `auto` uses the managed conda runner with ANTs on Linux and
the native SimpleITK route on macOS and Windows, and the native route is experimental because it
did not match the managed route closely enough on 2 of 24 test cases (the parity rule). It is saved as
`run_defaults.pnl_mask_route`; the mask receipt and the run record state the route.

On Linux x86_64, **Environment and installation > Prepare selected
analysis environment** can install the pinned PNL runtime (a conda environment
with ANTs 2.3.0), weights and template. The licence appears in the one setup
dialog. The setup then fills all four paths. The managed route runs the PNL
inference code, which this plugin also carries as a modified copy for the native
route (see the [PNL licence page](../licenses/pnl_brain_mask.md)); it is not a rewrite of PNLpipeline.
On macOS and Windows no managed runtime exists: the native route runs in Slicer's
own Python and needs only the PyTorch extension and the pnl-cnn assets.
See [managed setup](MANAGED_SETUP.md) for repair, verification and platform limits.
For an existing managed environment, configure **External Runtime Paths**
(the four PNL paths are read only by the managed route):

| Setting | Browse Mode |
| --- | --- |
| `pnl_mask_python` | Python executable file in the user's PNL environment |
| `pnl_mask_script` | The `pnl-brain-mask` console script of the installed PNLBrainMask-Torch package |
| `pnl_mask_model_dir` | Model directory |
| `pnl_mask_ants_bin_dir` | ANTs binary directory |

The SynthStrip, MRtrix `dwi2mask` and DIPY mask backends were removed. A saved setting with one of
them loads as `pnl_cnn` with one note in the log.
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
default, Advanced placement, both choices, node/UI synchronization,
debounced saves, EN/ZH labels, external-path persistence, and restoration in a
new process. Its report and screenshots are written under `--report`.

---

## 中文

脑掩膜选择器仅位于“高级设置 > 预处理与掩膜 > 脑掩膜方法”，默认显示
`PNL CNN (dMRI)`。另一选项是“仅使用提供的掩膜”。SynthStrip、MRtrix dwi2mask 和 DIPY median-Otsu 已移除；保存的旧设置会改为 `pnl_cnn` 并在日志中提示一次。
已有的用户掩膜会被尊重；依赖或掩膜缺失时明确报错，不会自动换用其他后端。
选择默认方法不代表 PNL 模型或运行环境已安装。

旁边的“掩膜计算设备”默认 `auto`，PNL CNN 优先使用其外部
Python 中可见的第一个 CUDA GPU；`cuda` 要求 GPU 可用，`cpu` 显式使用 CPU。
选择会持久化并传给单被试、批处理和预处理流程。GPU 报错不会静默改用 CPU。
详见[GPU 策略](GPU_EXECUTION.zh-CN.md)。

高级选项“PNL 路线”（仅在选择 PNL CNN 时显示）可选 `auto`、`managed` 或 `native（实验性）`：
`auto` 在 Linux 使用带 ANTs 的 managed conda 运行环境，在 macOS 和 Windows 使用 native SimpleITK 路线；
native 路线为实验性，因为它在 M-A 一致性规则的 24 例中有 2 例未达标（T2 设计 §3）。
选择保存为 `run_defaults.pnl_mask_route`，掩膜回执与运行记录会写明所用路线。

Linux x86_64 可点击“环境与安装 > 准备所选分析环境”，确认条款后自动准备
固定版本的官方 PNL Python 环境、权重与模板，校验后填写路径；并非重写 PNLpipeline。
修复、验证和平台限制见[自动配置](MANAGED_SETUP.zh-CN.md)。
已有环境仍可在“外部运行路径”配置用户自己的 PNL Python
可执行文件、官方 `dwi_masking.py`、模型目录及 ANTs bin 目录。选择和路径会写入共享用户 JSON，供 GUI 与 CLI
复用；上述运行时测试隔离用户设置，并通过两次启动检查持久化。
