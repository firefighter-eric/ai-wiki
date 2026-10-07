---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wang et al. - 2021 - LayoutReader Pre-training of Text and Layout for Reading Order Detection

## TL;DR（快速导读）

LayoutReader 利用文字和版面判断阅读顺序，并从 Word 文件的结构信息自动构造训练数据。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

复杂页面中，位置从左到右不一定就是正确阅读顺序。论文用文字与布局共同建模，并从文档元数据获得较大规模监督。它处理的是排序问题，仍需与字符识别和结构恢复协同。

## 具体怎么理解

一页有正文、侧栏和图注时，顺着坐标排序可能把无关内容插进句子；正确顺序应尊重段落与区域。

## 关键事实

- **C1**：LayoutReader 联合文字与布局，用序列到序列方式预测阅读顺序。
- **C2**：ReadingBank 提供约五十万文档图像的阅读顺序数据；论文还评估输入排列和 OCR 适配。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Wang%20et%20al.%20-%202021%20-%20LayoutReader%20Pre-training%20of%20Text%20and%20Layout%20for%20Reading%20Order%20Detection.pdf)
- 全文文本：[打开全文文本](../../raw/text/Wang%20et%20al.%20-%202021%20-%20LayoutReader%20Pre-training%20of%20Text%20and%20Layout%20for%20Reading%20Order%20Detection.md)
- 作者：Wang et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Wang%20et%20al.%20-%202021%20-%20LayoutReader%20Pre-training%20of%20Text%20and%20Layout%20for%20Reading%20Order%20Detection.html)

## 争议与不确定点

- 训练文档结构与真实 OCR 噪声不同，需要检查适配结果。
- 排序无法修复已丢失的文字或错误识别。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

阅读顺序错误会把双栏论文或表格内容拼乱。LayoutReader 利用语义与坐标共同恢复顺序，说明 OCR 之后仍有独立的结构重建工作。接入本库时，文本数量充足不能替代顺序检查。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202021%20-%20LayoutReader%20Pre-training%20of%20Text%20and%20Layout%20for%20Reading%20Order%20Detection.md#source-section-10 ) | 输入已有文字块，不是从像素直接识字 |
| C2 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202021%20-%20LayoutReader%20Pre-training%20of%20Text%20and%20Layout%20for%20Reading%20Order%20Detection.md#source-section-27 ) | 顺序准确性与 OCR 字符准确性是不同任务 |

## 核证范围

核对 §4 模型、§5 评测任务与 §7 数据规模说明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。
