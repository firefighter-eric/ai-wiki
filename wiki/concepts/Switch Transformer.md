---
type: concept
---
# Switch Transformer

## TL;DR（快速导读）

Switch Transformer 简化专家路由，让每个输入只使用少量专家，研究扩大容量时如何控制计算和训练不稳定。

## 简介

Switch Transformer 简化专家路由，让每个输入只使用少量专家，研究扩大容量时如何控制计算和训练不稳定。

## 具体怎么理解

系统保存多个专家，但单次输入只走选中的分支；专家负载与跨设备通信仍影响速度。

## 关键属性

- 类型：MoE 预训练模型
- 代表来源：[Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](../../wiki/summaries/Fedus,%20Zoph,%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.md)
- 当前角色：MoE 主线中的经典代表

## 相关主张

- Switch Transformer 展示了稀疏激活对参数扩展和训练效率的作用。
- 在当前知识库里，它是理解 DeepSeek-V3 这类后续 MoE 模型的早期参照。

## 来源支持

- [Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](../../wiki/summaries/Fedus,%20Zoph,%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.md)

## 关联页面

- [MoE](./MoE.md)
- [T5](./T5.md)
- [DeepSeek-V3](./DeepSeek-V3.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

## 这里的术语是什么意思

- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
