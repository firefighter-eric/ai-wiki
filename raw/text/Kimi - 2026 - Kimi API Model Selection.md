# Kimi API Model Selection

- Source HTML: `raw/html/Kimi - 2026 - Kimi API Model Selection.html`
- Source URL: https://www.kimi.com/help/kimi-api/api-model-selection
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

- ProductsProducts

KimiAll-in-one agentic AI workspace

Kimi WorkAI desktop agent for knowledge workers

Kimi CodeAI code agent for terminal & IDE

Kimi WebBridgeA browser extension for AI agents

Kimi PlatformAccess the latest Kimi models

- FeaturesFeatures

SlidesAI presentation maker

WebsitesAI website builder

Deep ResearchGet thorough & multi-format reports

SheetsBuild Excel formulas, pivots & charts

DocsCreate, convert & review documents

Kimi ClawDeploy 24/7 AI agents in one click

- ResearchResearch

Kimi K3Open Frontier Intelligence

PerceptionBenchAtomic Visual Perception in MLLMs

Kimi K2.6Advancing Open-Source Coding

Agent SwarmScale Out, Not Just Up

WorldVQAAtomic World Knowledge in MLLMs

Kimi K2.5Visual Agentic Intelligence

Kimi Vendor VerifierRebuilding the Chain of Trust

Kimi K2 ThinkingOpen-source thinking model

Kimi K2Open Agentic Intelligence

- ResourcesResources

Kimi Code Introduction

Parallel Agent

Multi Agent

Hermes Agent Overview

Hermes API Integration

OpenClaw SaaS

How to Install OpenClaw on Mac

AI Tools for Excel

Vibe Coding Guide

How to Vibe Code

How to Build a Website from Scratch

Refactor moonshot.ai with Kimi Code CLI

Kimi AI Examples and Showcases

- PricingPricing

- HelpHelp

- Kimi K3Kimi K3

LoginTry Kimi

- Getting Started

- Features

- Agent Mode

- Kimi Work

- Kimi WebBridge

- Slides

- Docs & Sheets

- Deep Research

- Websites

- Membership

- Kimi Code

- Kimi API

- Kimi API overview

- Error codes

- API pricing

- Balance & usage

- Rate limits

- Model selection

- API troubleshooting

- Account & authentication

- Billing & finance

- Model capabilities

- Business cooperation & sales

- Data processing & security

- Kimi Business

- Kimi Claw

- Kimi Bulletin

- FAQ

Help CenterKimi APIModel selection

# Model selection

Kimi API provides a variety of models for developers to choose from. Different models have different strengths in capability, speed, and price.

## Main models

### kimi-k3

kimi-k3 is Kimi's flagship model, built for long-horizon coding and end-to-end knowledge work, with native visual understanding. K3 always runs in thinking mode. You can set the thinking effort via the top-level reasoning_effort parameter (supports low, high, max; defaults to max). It supports a context window of up to 1M tokens.

### kimi-k2.6

kimi-k2.6 supports text, image, and video input, and supports switching between thinking and non-thinking modes. It is suitable for conversation, code generation, visual understanding, and agent tasks. It supports a context window of up to 256k tokens.

For the full list of available models and their parameters, visit platform.kimi.ai/docs/introduction.

## Selection dimensions

When choosing a model, evaluate the following dimensions:

- Context length: Different models support different maximum context windows. Choose a larger-context model for long-document processing.

- Response speed: Lighter models respond faster, making them suitable for latency-sensitive scenarios.

- Generation quality: Higher-tier models perform better on complex reasoning and creative tasks.

- Price: Choose the most cost-effective model based on your budget and call volume.

- Thinking mode: Choose kimi-k3 for deep reasoning, or kimi-k2.6 when you need flexibility to switch thinking mode on and off.

## Vision models (image understanding)

Vision models support image input and can be used for image description, OCR, chart interpretation, and similar tasks:

- Each image is billed at a fixed 1024 tokens, regardless of image size or resolution.

- Common image formats are supported (JPEG, PNG, WebP, etc.).

- Images can be passed via URL or Base64 encoding.

## Currently unsupported capabilities

- PPT generation API: PPT generation is not currently available via API.

- Deep research API: Deep research is not currently available via API.

Watch the platform announcements for the latest updates.

Was this article helpful?

YesNo

PreviousRate limitsNextAPI troubleshooting

- Main models

- kimi-k3

- kimi-k2.6

- Selection dimensions

- Vision models (image understanding)

- Currently unsupported capabilities

#### Products

#### Features

#### Capabilities

#### Company

English
