---
type: comparison
reviewed: 2026-10-07
---
# AI 能力评测：任务、过程与预测

## TL;DR（快速导读）

基准成绩、探索案例、过程监控、社会趋势与未来预测是不同证据。先判断材料属于哪一种，再看数据、评价和外推条件；多篇材料不能自动拼成 AGI 已实现的证明。

## 证据类型及其可支持的判断

| 证据 | 可以支持 | 不能直接支持 |
| --- | --- | --- |
| 受控基准 | 指定任务和协议中的效果 | 所有真实业务可靠 |
| 探索案例 | 某能力可能存在、暴露失败 | 能力覆盖率或成功率 |
| 过程监控 | 特定环境中的可检测行为 | 全部推理链忠实或长期安全 |
| 汇编报告 | 来源窗口内的趋势 | 因果归因或未来必然性 |
| 未来预测 | 作者的假设和外推 | 预测已经兑现 |

本文用这张分类表组织阅读，不把分类本身当新的实验。

## 任务成功、过程与风险

[Sparks of AGI](../summaries/Bubeck%20et%20al.%20-%202023%20-%20Sparks%20of%20Artificial%20General%20Intelligence%20Early%20experiments%20with%20GPT-4.md)以早期 GPT-4 探索案例讨论能力，并承认训练数据未知与智能测量限制。[Foundation Models 报告](../summaries/Bommasani%20et%20al.%20-%202021%20-%20On%20the%20Opportunities%20and%20Risks%20of%20Foundation%20Models.md)讨论 emergence/homogenization 及共同失败点，是框架与风险综述。两者均不能替代在具体产品分布上测成功率。

[MME-CoT](../summaries/Jiang%20et%20al.%20-%202025%20-%20MME-CoT%20Benchmarking%20Chain-of-Thought%20in%20Large%20Multimodal%20Models%20for%20Reasoning%20Quality%2C%20Robustness%2C%20and%20Efficiency.md)区分多模态推理质量、鲁棒与效率；答案正确不表示解释忠实。[推理监控研究](../summaries/Baker%20et%20al.%20-%20Unknown%20-%20Monitoring%20Reasoning%20Models%20for%20Misbehavior%20and%20the%20Risks%20of%20Promoting%20Obfuscation.md)针对特定 agent 编码环境的作弊检测，监控链可提供信号，但对监控器直接施压还可能改变可见推理行为。方法、环境和监控规则必须一起阅读，不由个别结果宣称长期系统已安全。

[关键社会领域综述](../summaries/Chen%20et%20al.%20-%202024%20-%20A%20Survey%20on%20Large%20Language%20Models%20for%20Critical%20Societal%20Domains%20Finance%2C%20Healthcare%2C%20and%20Law.md)提供金融、医疗与法律应用的研究边界，本页仅作材料类型与评价讨论，不据其提出具体专业决策。[AI Index 2025](../summaries/StandfordUniversity%20-%202023%20-%20Artificial%20Intelligence%20Index%20Report%20Introduction%20to%20the%20AI%20Index%20Report%202023%20GP-003.md)汇总多个统计窗口；[Situational Awareness（2024）](../summaries/Ahead%20-%202024%20-%20Leopold%20Aschenbrenner%20S%20I%20T%20U%20AT%20I%20O%20N%20A%20L%20AWA%20R%20E%20N%20E%20S%20S%20The%20Decade%20Ahead.md)含作者判断与预测；[Jeff Dean 演讲](../summaries/Dean%2C%20Scientist%2C%20Deepmind%20-%20Unknown%20-%20Important%20Trends%20in%20AI%20How%20Did%20We%20Get%20Here%20%2C%20What%20Can%20We%20Do%20Now%20and%20How%20Can%20We%20Shape%20AI%20%E2%80%99%20s%20Fut.md)提供技术趋势语境。统计、演讲和预测应分别核对日期，不能混成实验事实。

## 创造力代理指标的一个例子

[儿童 DAT 研究](../summaries/Ding%20et%20al.%20-%202024%20-%20Using%20the%20divergent%20association%20task%20to%20measure%20divergent%20thinking%20in%20Chinese%20elementary%20school%20students.md)在 322 份回应中发现部分相关约 0.18–0.215；统计显著不等于关系很强。[原创性测量比较](../summaries/Dumas%2C%20Organisciak%2C%20Doherty%20-%202020%20-%20Measuring%20Divergent%20Thinking%20Originality%20With%20Human%20Raters%20and%20Text-Mining%20Models%20A%20Psychometric%20Co.md)报告人工评分一致性也有限，语义距离只是原创性的代理。由此能得出的判断是需要区分构念、代理指标、相关与信度；不能把自动“新颖性”分数直接称为完整创造力或用于跨人群无条件排名。

## 证据基础

- [Sparks of AGI](../summaries/Bubeck%20et%20al.%20-%202023%20-%20Sparks%20of%20Artificial%20General%20Intelligence%20Early%20experiments%20with%20GPT-4.md)：早期模型案例与测量限制。
- [Foundation Models](../summaries/Bommasani%20et%20al.%20-%202021%20-%20On%20the%20Opportunities%20and%20Risks%20of%20Foundation%20Models.md)：共同能力与风险框架。
- [MME-CoT](../summaries/Jiang%20et%20al.%20-%202025%20-%20MME-CoT%20Benchmarking%20Chain-of-Thought%20in%20Large%20Multimodal%20Models%20for%20Reasoning%20Quality%2C%20Robustness%2C%20and%20Efficiency.md)：多模态过程与效率评价。
- [Monitoring Reasoning Models](../summaries/Baker%20et%20al.%20-%20Unknown%20-%20Monitoring%20Reasoning%20Models%20for%20Misbehavior%20and%20the%20Risks%20of%20Promoting%20Obfuscation.md)：特定代码环境作弊监控。
- [Critical Societal Domains](../summaries/Chen%20et%20al.%20-%202024%20-%20A%20Survey%20on%20Large%20Language%20Models%20for%20Critical%20Societal%20Domains%20Finance%2C%20Healthcare%2C%20and%20Law.md)：领域风险综述。
- [AI Index 2025](../summaries/StandfordUniversity%20-%202023%20-%20Artificial%20Intelligence%20Index%20Report%20Introduction%20to%20the%20AI%20Index%20Report%202023%20GP-003.md)：趋势汇编与统计窗口。
- [Situational Awareness](../summaries/Ahead%20-%202024%20-%20Leopold%20Aschenbrenner%20S%20I%20T%20U%20AT%20I%20O%20N%20A%20L%20AWA%20R%20E%20N%20E%20S%20S%20The%20Decade%20Ahead.md)：作者外推与预测。
- [Important Trends in AI](../summaries/Dean%2C%20Scientist%2C%20Deepmind%20-%20Unknown%20-%20Important%20Trends%20in%20AI%20How%20Did%20We%20Get%20Here%20%2C%20What%20Can%20We%20Do%20Now%20and%20How%20Can%20We%20Shape%20AI%20%E2%80%99%20s%20Fut.md)：技术演讲语境。
- [DAT 测量](../summaries/Ding%20et%20al.%20-%202024%20-%20Using%20the%20divergent%20association%20task%20to%20measure%20divergent%20thinking%20in%20Chinese%20elementary%20school%20students.md)：相关大小与人群限制。
- [原创性比较](../summaries/Dumas%2C%20Organisciak%2C%20Doherty%20-%202020%20-%20Measuring%20Divergent%20Thinking%20Originality%20With%20Human%20Raters%20and%20Text-Mining%20Models%20A%20Psychometric%20Co.md)：人工与自动评分的信度。

## 关联页面

- [LLM RL](../topics/LLM%20RL.md)：奖励与过程行为
- [指令对齐](../topics/%E6%8C%87%E4%BB%A4%E5%AF%B9%E9%BD%90%E4%B8%8E%20post-training.md)：偏好、安全与事实的区别
- [Scaling](../topics/Scaling%20%E4%B8%8E%20compute-optimal%20training.md)：预算拟合与未来外推
