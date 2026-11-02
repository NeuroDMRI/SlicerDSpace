<div align="center">

# Python 工作流 API

<p align="center"><a href="PYTHON_API.md">English</a> · <a href="CLI_TEMPLATES.zh-CN.md">CLI 模板</a></p>

</div>

---

在准备好的执行环境中，使用带预检的 `run_subject_workflow` 入口：

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

入口先检查所选方法的软件包、命令和资源，保存环境报告，再调用共用执行引擎。预检执行一次，不自动安装依赖。缺失资源可通过模块的环境准备入口或 [Agent Skill](../../.agents/skills/slicerdspace/SKILL.md) 配置。

`strict_environment=True` 在所选任务缺少依赖时停止执行。默认允许部分执行并记录受阻任务；`execute=False` 生成运行计划。输入校验与产物 QC 仍在执行流程中进行。

原有 `run_subject_products` 保留为兼容的低层入口，不执行全局环境预检。CLI 自行预检后调用该引擎。已对相同配置完成检查的高级调用方，可显式设置 `run_subject_workflow(..., check_environment=False)`。

[English](PYTHON_API.md) · [CLI 模板](CLI_TEMPLATES.zh-CN.md)
