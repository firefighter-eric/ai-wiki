# Docling Document Processing Overview

- Source HTML: `raw/html/Docling Project - 2026 - Document Processing Overview.html`
- Source SHA256: `9c6110a4d6e0236ffdb3841873ddfd192a5176bfd3c18cb601441dffeb6eab31`
- Source URL: https://docling-project.github.io/docling/
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

<a id="source-section-1"></a>

# Index

[图片：Docling] 
 [[图片：DS4SD%2Fdocling | Trendshift]](https://trendshift.io/repositories/17240)

[[图片：arXiv]](https://arxiv.org/abs/2408.09869)
[[图片：PyPI version]](https://pypi.org/project/docling/)
[[图片：PyPI - Python Version]](https://pypi.org/project/docling/)
[[图片：uv]](https://github.com/astral-sh/uv)
[[图片：Ruff]](https://github.com/astral-sh/ruff)
[[图片：Pydantic v2]](https://pydantic.dev)
[[图片：prek]](https://pypi.org/project/prek/)
[[图片：License MIT]](https://opensource.org/licenses/MIT)
[[图片：PyPI Downloads]](https://pepy.tech/projects/docling)
[[图片：Docling Actor]](https://apify.com/vancura/docling)
[[图片：Chat with Dosu]](https://app.dosu.dev/097760a8-135e-4789-8234-90c8837d7f1c/ask?utm_source=github)
[[图片：Discord]](https://docling.ai/discord)
[[图片：OpenSSF Best Practices]](https://www.bestpractices.dev/projects/10101)
[[图片：LF AI & Data]](https://lfaidata.foundation/projects/)

Docling simplifies document processing by parsing diverse formats — including advanced PDF understanding — and providing seamless integrations with the generative AI ecosystem.

<a id="source-section-2"></a>

## Getting started

🐣 Ready to kick off your Docling journey? Let's dive right into it!

[**⬇️ Installation**
Quickly install Docling in your environment](https://docling-project.github.io/docling/getting_started/installation/)
 [**▶️ Quickstart**
Get a jumpstart on basic Docling usage](https://docling-project.github.io/docling/getting_started/quickstart/)
 [**🧩 Concepts**
Learn Docling fundamentals and get a glimpse under the hood](https://docling-project.github.io/docling/concepts/)
 [**🧑🏽‍🍳 Examples**
Try out recipes for various use cases, including conversion, RAG, and more](https://docling-project.github.io/docling/examples/)
 [**🤖 Integrations**
Check out integrations with popular AI tools and frameworks](https://docling-project.github.io/docling/integrations/)
 [**📖 Reference**
See more API details](https://docling-project.github.io/docling/reference/document_converter/)

<a id="source-section-3"></a>

## Features

- 🗂️ Parsing of [multiple document formats](https://docling-project.github.io/docling/usage/supported_formats/) including PDF, DOCX, PPTX, XLSX, HTML, EPUB, Apple Pages, Numbers & Keynote, WAV, MP3, WebVTT, Box Notes, email formats (EML, MSG), images (PNG, TIFF, JPEG, ...), LaTeX, DocLang, plain text, and more

- 📑 Advanced PDF understanding incl. page layout, reading order, table structure, code, formulas, image classification, and more

- 🧬 A unified, expressive [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/) representation format

- ↪️ Various [export formats](https://docling-project.github.io/docling/usage/supported_formats/) and options, including Markdown, HTML, WebVTT, DocLang, [DocTags](https://arxiv.org/abs/2503.11576) and lossless JSON

- 📜 Support for several application-specific XML schemas including [DocLang](https://doclang.ai), [USPTO](https://www.uspto.gov/patents) patents, [JATS](https://jats.nlm.nih.gov/) articles, and [XBRL](https://www.xbrl.org/) financial reports.

- 🔒 Local execution capabilities for sensitive data and air-gapped environments

- 🤖 Plug-and-play [integrations](https://docling-project.github.io/docling/integrations/) incl. LangChain, LlamaIndex, Crew AI & Haystack for agentic AI

- 🔍 Extensive [OCR support](https://docling-project.github.io/docling/concepts/OCR/) for scanned PDFs and images

- 👓 Support for several Visual Language Models, such as ([GraniteDocling](https://huggingface.co/ibm-granite/granite-docling-258M))

- 🎙️ Audio support with Automatic Speech Recognition (ASR) models

- 🔌 Connect to any agent using the [MCP server](https://docling-project.github.io/docling/usage/mcp/)

- 🌐 Run Docling as a service with the [API server](https://docling-project.github.io/docling/usage/api_server/) (docling-serve)

- 💻 Simple and convenient CLI

<a id="source-section-4"></a>

### What's new

- 🎬 Parsing of video files (MP4, AVI, MOV, MKV, and WebM) with an ASR transcript and representative keyframes

- 📄 Parsing of ODF (OpenDocument Format) files for text documents (`.odt`), spreadsheets (`.ods`), and presentations (`.odp`)

- 💼 Parsing of XBRL (eXtensible Business Reporting Language) documents for financial reports

- 📧 Parsing of email files (`.eml`, `.msg`)

- 📚 Parsing of EPUB (Electronic Publication) files for e-books

- 🍎 Parsing of Apple Pages (`.pages`) documents, Numbers (`.numbers`) spreadsheets and Keynote (`.key`) presentations

- 📝 Parsing of plain-text files (`.txt`, `.text`) and Markdown supersets (`.qmd`, `.Rmd`)

- 📊 Chart understanding (Barchart, Piechart, LinePlot): convert them into tables or code and add detailed descriptions

- 🔠 Opt-in recovery of [PDF heading levels](https://docling-project.github.io/docling/usage/heading_levels/) from the bookmarks, the outline numbering and the font styling, instead of a flat list of level-1 headings

<a id="source-section-5"></a>

### Coming soon

- 📝 Metadata extraction, including title, authors, references & language

- 📝 Complex chemistry understanding (Molecular structures)

<a id="source-section-6"></a>

## What's next

🚀 The journey has just begun! Join us and become a part of the growing Docling community.

- [GitHub](https://github.com/docling-project/docling)

- [Discord](https://docling.ai/discord)

- [LinkedIn](https://linkedin.com/company/docling/)

<a id="source-section-7"></a>

## Live assistant

Do you want to leverage the power of AI and get live support on Docling?
Try out the [Chat with Dosu](https://app.dosu.dev/097760a8-135e-4789-8234-90c8837d7f1c/ask?utm_source=github) functionalities provided by our friends at [Dosu](https://dosu.dev/).

[[图片：Chat with Dosu]](https://app.dosu.dev/097760a8-135e-4789-8234-90c8837d7f1c/ask?utm_source=github)

<a id="source-section-8"></a>

## LF AI & Data

Docling is hosted as a project in the [LF AI & Data Foundation](https://lfaidata.foundation/projects/).

<a id="source-section-9"></a>

### IBM ❤️ Open Source AI

The project was started by the AI for knowledge team at IBM Research Zurich.
