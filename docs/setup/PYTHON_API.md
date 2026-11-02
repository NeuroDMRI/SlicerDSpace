<div align="center">

# Python Workflow API

<p align="center"><a href="PYTHON_API.zh-CN.md">中文</a> · <a href="CLI_TEMPLATES.md">CLI templates</a></p>

</div>

---

Use `run_subject_workflow` for a checked Python entry point in a prepared execution environment:

```python
from pathlib import Path
from shared import ProductOrchestratorConfig, run_subject_workflow

result = run_subject_workflow(
    ProductOrchestratorConfig(
        subject_id="sub-01",
        subject_root=Path("/path/to/subject"),
        output_root=Path("/path/to/results"),
        targets=("microstructure",),
        execute=True,
    ),
    strict_environment=True,
    log=print,
)
```

The entry point checks the selected methods' packages, commands and assets once, writes an environment report, then invokes the shared execution engine. It does not install dependencies. Prepare missing resources through the module's preparation workflow or the [Agent Skill](../../.agents/skills/slicerdspace/SKILL.md).

`strict_environment=True` stops execution if selected targets have missing dependencies. The default permits partial execution and records blocked targets. `execute=False` creates a plan. Input validation and product QC remain part of execution.

The existing `run_subject_products` remains a low-level compatible entry without the global environment check. The CLI performs its own check before invoking that engine. Advanced callers that have already checked the same configuration can explicitly use `check_environment=False` with `run_subject_workflow`.

[中文](PYTHON_API.zh-CN.md) · [CLI templates](CLI_TEMPLATES.md)
