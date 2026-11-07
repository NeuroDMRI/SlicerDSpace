<div align="center">

# Shared User Configuration

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

SlicerDSpace stores common preferences in one local JSON file shared by the
Slicer GUI and command-line interface. After the first setup, users do not need
to repeatedly enter language, model paths, external software paths or common
run defaults.

---

## Default Location

| Platform | Default JSON path |
| --- | --- |
| Windows | `%APPDATA%\SlicerDSpace\user_config.json` |
| macOS | `~/Library/Application Support/SlicerDSpace/user_config.json` |
| Linux/HPC | `$XDG_CONFIG_HOME/SlicerDSpace/user_config.json` or `~/.config/SlicerDSpace/user_config.json` |

Set `SLICERDSPACE_USER_CONFIG=/custom/path/user_config.json` to use another
location. This is useful for a lab-wide profile, portable test environment or
CI job.

---

## Saved Values

- UI language and whether the first-run wizard has been shown.
- Recent subject ID, subject root and output directory.
- Default targets, run name, preprocessing/tractography/bundle/connectome
  backends, microstructure stages and streamline count.
- Python environment, model asset root, MRtrix3, FSL, FreeSurfer, TractQuerier,
  WMA, Slicer and TractCloud paths.
- The PNL brain-mask route (`run_defaults.pnl_mask_route`: `auto`, `managed` or
  `native`; an unknown value is read as `auto`).
- The licences you accepted in the setup dialog (`accepted_licences`: id, version
  and time), so a licence that has not changed is not asked for again.

The GUI still writes compatible Slicer QSettings. The JSON file is the portable
configuration that can be inspected, copied, versioned privately or reused by
the CLI. Common GUI edits are saved automatically after a short debounce;
language and dependency-path changes are saved immediately. Do not publish a
config containing private data paths without review.

---

## Language

New profiles default to English. The `Language` selector switches the running
module between English and Chinese and saves the choice immediately. The same
default can be managed directly in JSON:

```json
{
  "ui": {
    "language": "en"
  }
}
```

Use `"en"` for English or `"zh"` for Chinese. If the file is edited while the
module is already open, reload the module or restart Slicer so the edited value
is read again. Dependency installation merges its paths into this file without
overwriting an existing language choice.

---

## CLI Use

Print the automatically loaded configuration:

```bash
PythonSlicer run_subject_pipeline.py --print-user-config
```

Save resolved CLI choices to the default JSON path:

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /data/project/sub-01 \
  --targets validation,microstructure,tractography \
  --tractography-backend dti_det \
  --save-user-config \
  --doctor
```

Reuse the saved subject root and defaults:

```bash
PythonSlicer run_subject_pipeline.py --doctor
```

Use an isolated test profile:

```bash
PythonSlicer run_subject_pipeline.py \
  --user-config /tmp/slicerdsurfer-test-user-config.json \
  --subject-root /data/project/sub-01 \
  --targets validation \
  --save-user-config /tmp/slicerdsurfer-test-user-config.json \
  --doctor
```

Disable saved settings for one command:

```bash
PythonSlicer run_subject_pipeline.py \
  --no-user-config \
  --subject-root /data/project/sub-01 \
  --targets validation \
  --doctor
```

Precedence is: direct CLI option, explicit `--dependency-config` profile,
shared user config, then built-in default. This lets a reusable project profile
override personal dependency paths without overriding an option typed directly
for the current command.
When saving, SlicerDSpace records the final resolved target and
microstructure-stage selections after applying `--all`, `--skip-targets` and
stage expansion. GUI-only choices, such as the first-run state and automatic
Python-package preference, are preserved.
