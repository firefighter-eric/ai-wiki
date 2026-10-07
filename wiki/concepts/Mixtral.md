---
type: concept
---
# Mixtral

## TL;DR（快速导读）

Mixtral 将语言模型组织为专家混合结构，每次只计算部分专家，以更高总容量控制单次计算。

## 简介

Mixtral 将语言模型组织为专家混合结构，每次只计算部分专家，以更高总容量控制单次计算。

## 具体怎么理解

保存全部专家仍需要内存，多卡部署还可能产生通信；只看激活参数量会漏掉这些成本。

## 关键属性

- 类型：开放语言模型 / `MoE` 家族节点
- 开放性：`open-weight`
- 代表来源：[Jiang et al. - 2024 - Mixtral of Experts](../../wiki/summaries/Jiang%20et%20al.%20-%202024%20-%20Mixtral%20of%20Experts.md)
- 当前角色：开放 `MoE` 模型家族的代表节点

## 相关主张

- `Mixtral` 把 Mistral 的高效路线延伸到稀疏激活模型。
- 它说明开放模型竞争已经不再局限于 dense scaling，而是进入 `MoE` 工程主线。
- 在当前知识库里，`Mixtral` 适合作为 `MoE` 家族在开放模型中的具体落地案例。

## 来源支持

- [Jiang et al. - 2024 - Mixtral of Experts](../../wiki/summaries/Jiang%20et%20al.%20-%202024%20-%20Mixtral%20of%20Experts.md)

## 关联页面

- [Mistral 7B](./Mistral 7B.md)
- [MoE](./MoE.md)
- [DeepSeek-V3](./DeepSeek-V3.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)

## 这里的术语是什么意思

- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
