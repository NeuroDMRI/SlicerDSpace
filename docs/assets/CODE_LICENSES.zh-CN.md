<div align="center">

# 代码与第三方授权

**许可证原文、来源与分发边界**

[项目](../../README.zh-CN.md) | [文档](../README.zh-CN.md) | [模型资源](MODEL_ASSET_TERMS.md) | [发布门禁](../release/EXTENSION_READINESS.zh-CN.md) | [English](../../THIRD_PARTY_NOTICES.md)

</div>

根目录只保留 [LICENSE.txt](../../LICENSE.txt)，内容为项目自身代码的
BSD-3-Clause 许可证。此前的 `LICENSE` 是同内容副本，并不是第二套授权；
已删除副本并更新安装清单、发布清单和链接。第三方许可证不能因此删除。

---

## 已补齐的声明

| 组件 | 原文与来源 | 分发条件与边界 |
| --- | --- | --- |
| pnlbwh/conversion 及 DDSurfer 内的派生转换工具 | [BWH Conversion 协议](../../anatomy/parcellation/conversion/LICENSE.txt)；[固定上游版本](https://github.com/pnlbwh/conversion/blob/cdd433adff3eebc486f2b5648714cbfe93b9422e/LICENSE) | 保留完整协议、归属声明，明确本地修改；不是项目 BSD-3-Clause 再授权 |
| FreeSurferColorLUT.txt | [MGH FreeSurfer 协议](../../licenses/FreeSurfer.txt)；[固定官方版本](https://github.com/freesurfer/freesurfer/blob/4f8728e5132e8d39a8f1bf3d24d15fed62f4fc9d/LICENSE.txt) | 按 Part B 保留条款和归属；LUT 标签与颜色不改；不代表模板或被试数据许可 |
| DDSurfer 自有源码 | [MIT 原文](../../anatomy/surface/ddsurfer_backend/LICENSE.txt)；[上游加入许可证的提交](https://github.com/NeuroDMRI/DDSurfer/blob/0c399ef8ff8c9931730cab6beaf5796ee61243ca/LICENSE) | 保留版权与许可；派生代码另遵守原许可证；不推断权重和模板许可 |
| PyTorch3D 派生 mesh 工具 | [Meta BSD-3-Clause 原文](../../licenses/PyTorch3D.txt)；[上游声明](https://github.com/NeuroDMRI/DDSurfer/blob/c3b2d40ead82b316f0adec900a23979f629da70b/utils/LICENSE.pytorch3d) | 保留 Meta 声明、条件与免责声明 |
| Free-water WLS 计算适配 | [DIPY BSD-3-Clause 原文](../../licenses/DIPY.txt)；[固定上游版本](https://github.com/dipy/dipy/blob/1b0b3e2a0b0c8611287f4677f21356b704495665/LICENSE) | 保留版权和完整条款；无对角矩阵的 WLS 是修改实现，不是新模型或作者背书 |
| WMA 批量筛选／点变换 | [WMA／Slicer BWH 协议](../../licenses/WMA.txt) | 保留完整协议及修改说明；不改变作者目标函数或优化器，不推断图谱许可 |

FastSurfer 的 Apache-2.0 和 SAF 的 MIT 声明继续保留。
MK-Curve 的 Python 移植算法保留 [BWH MK-Curve 协议](../../licenses/MK-Curve-LICENSE.txt)；
派生 DKI 张量计算与积分方向表保留 [MPL-2.0](../../licenses/MK-Curve-DKI-MPL-2.0.txt)。
这些文件作为修改后的 MPL 源码随插件分发，不将第三方代码改为项目 BSD 授权，
也不宣称为作者正式发布或背书的 Python 实现。
英文 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) 含归属声明与完整条款入口，
应随代码、安装包及其文档分发。本页是说明，不能替代原文。

---

## 校验与限制

~~~bash
python3 scripts/audit_third_party.py
~~~

[licenses/provenance.json](../../licenses/provenance.json) 记录固定提交、适用路径和
原文 SHA256。CI 会检查声明完整性、文件哈希、来源固定与安装包资源清单。

通过仅意味着这些记录已补齐，**不是法律认证，也不是所有资源可公开分发**。
模型权重、模板、初始化网格、被试衍生测试数据和相关权利仍需独立确认；
仓库继续保持私有，不创建官方 Index PR 或公开发布。
