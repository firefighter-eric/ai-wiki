# vLLM Architecture Overview

- Source HTML: `raw/html/vLLM Project - 2026 - Architecture Overview.html`
- Source URL: https://docs.vllm.ai/en/stable/design/arch_overview/
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

Skip to content

Initializing search

GitHub

- Home

- User Guide

- Developer Guide

- Benchmarking

- API Reference

- CLI Reference

- Community

GitHub

- Home

- Getting Started Getting Started

- Quickstart

- GPU

- CPU

- TPU

- Applications Applications

- API Server

- Chatbot

- Rag

- Basic Basic

- Offline Inference

- Online Serving

- Deployment Deployment

- Async LLM Streaming

- Helm Charts

- LLM Engine Example

- Sagemaker-Entrypoint

- Disaggregated Disaggregated

- Disaggregated Encoder

- Disaggregated Serving

- Ec Both Encoder

- Disaggregated Prefill V1

- Flexkv Connector

- KV Load Failure Recovery Test

- LMCache Examples

- Mooncake Connector

- Features Features

- Automatic Prefix Caching

- Batch Invariance

- Context Extension

- Data Parallel

- Kv Events

- Logging Configuration

- Custom Logits Processors

- LoRA

- Offline Inference with the OpenAI Batch file format

- Pause Resume

- Profiling

- Prompt Embed

- Reset Kv

- Sharded State

- Speculative Decoding

- Structured Outputs

- Tensorize vLLM Model

- Torchrun

- Generate Generate

- Batched Chat Completions Online

- Multimodal

- Qwen 1M Offline

- Observability Observability

- Monitoring Dashboards

- Metrics

- Setup OpenTelemetry POC

- Prometheus and Grafana

- Pooling Pooling

- Classify

- Embed

- Plugin

- Reward

- Score

- Token Classify

- Token Embed

- Ray Serving Ray Serving

- Batch LLM Inference

- Elastic Ep

- Multi-Node-Serving

- Ray Serve Deepseek

- Run Cluster

- Reasoning Reasoning

- OpenAI Chat Completion Tool Calls With Reasoning

- OpenAI Chat Completion With Reasoning

- OpenAI Chat Completion With Reasoning Streaming

- OpenAI Responses Client

- RL RL

- RLHF Async New APIs

- RLHF Http IPC

- RLHF Http NCCL

- RLHF IPC

- RLHF IPC Fsdp Ep

- RLHF NCCL

- RLHF NCCL Fsdp Ep

- RLHF Sparse NCCL

- Routed Experts E2E

- Skip Loading Weights In Engine Init

- Scale Out Scale Out

- Init

- Example Mm Serve

- Token Generation Client

- Speech To Text Speech To Text

- Lid

- OpenAI

- Realtime

- Tool Calling Tool Calling

- Chat With Tools Offline

- OpenAI Chat Completion Client With Tools

- OpenAI Chat Completion Client With Tools Required

- OpenAI Chat Completion Client With Tools Xlam

- OpenAI Chat Completion Client With Tools Xlam Streaming

- OpenAI Responses Client With Mcp Tools

- OpenAI Responses Client With Tools

- General General

- vLLM V1

- Frequently Asked Questions

- Production Metrics

- Reproducibility

- Security

- Troubleshooting

- Usage Stats Collection

- Inference and Serving Inference and Serving

- Offline Inference

- Derenderer APIs

- Generative Scoring

- OpenAI-Compatible Server

- Renderer APIs

- Speech to Text APIs

- Context Parallel Deployment

- Data Parallel Deployment

- Troubleshooting distributed deployments

- Expert Parallel Deployment

- Parallelism and Scaling

- Integrations Integrations

- Claude Code

- Codex

- LangChain

- LlamaIndex

- Deployment Deployment

- Using Docker

- Using Kubernetes

- Using Nginx

- Frameworks Frameworks

- Anyscale

- AnythingLLM

- AutoGen

- BentoML

- Cerebrium

- Chatbox

- Dify

- dstack

- Haystack

- Helm

- Hugging Face Inference Endpoints

- LiteLLM

- Lobe Chat

- LWS

- Modal

- Open WebUI

- Retrieval-Augmented Generation

- RunPod

- SkyPilot

- Streamlit

- NVIDIA Triton

- Integrations Integrations

- AIBrix

- NVIDIA Dynamo

- KAITO

- KServe

- Kthena

- KubeAI

- KubeRay

- Llama Stack

- llm-d

- llmaz

- Production stack

- Training Training

- Async Reinforcement Learning

- What is Layerwise (Re)loading?

- Reinforcement Learning from Human Feedback

- Transformers Reinforcement Learning

- Base Class and Custom Engines

- IPC Engine

- NCCL Engine

- Conserving Memory

- Engine Arguments

- Environment Variables

- Model Resolution

- Optimization and Tuning

- Server Arguments

- TPU

- Models Models

- Supported Models

- Generative Models

- Classification Usages

- Embedding Usages

- Reward Usages

- Scoring Usages

- Specific Model Examples

- Token Classification Usages

- Token Embedding Usages

- Extensions Extensions

- Loading model weights with fastsafetensors

- Loading Model Weights with InstantTensor

- Loading models with Run:ai Model Streamer

- Loading models with CoreWeave's Tensorizer

- Hardware Supported Models Hardware Supported Models

- CPU - Intel® Xeon®

- XPU - Intel® GPUs

- TPU

- Automatic Prefix Caching

- Batch Invariance

- Context Extension

- Custom Arguments

- Custom Logits Processors

- Disaggregated Encoder

- Disaggregated Prefilling (experimental)

- IndexCache

- Interleaved Thinking

- KV Offloading Usage Guide

- LoRA Adapters

- MooncakeConnector Usage Guide

- MooncakeStoreConnector Usage Guide

- MoRIIOConnector Usage Guide

- Multimodal Inputs

- NixlConnector Compatibility Matrix

- NixlConnector Usage Guide

- Per-Request Metrics

- Prompt Embedding Inputs

- Reasoning Outputs

- Sleep Mode

- Structured Outputs

- Tool Calling

- AutoAWQ

- BitsAndBytes

- FP8 ViT Encoder Attention

- GGUF

- GPTQModel

- Intel Quantization Support

- NVIDIA Model Optimizer

- Online Quantization

- Quantized KV Cache

- AMD Quark

- TorchAO

- FP8 W8A8

- INT4 W4A16

- INT8 W4A8

- INT8 W8A8

- Draft Models

- Dynamic Speculative Decoding

- EAGLE Draft Models

- Hidden State Extraction

- MLP Draft Models

- MTP (Multi-Token Prediction)

- N-Gram Speculation

- Parallel Draft Models

- vLLM-Project/Speculators

- Suffix Decoding

- General General

- Deprecation Policy

- Dockerfile

- Editing Agent Instructions

- Incremental Compilation Workflow

- Profiling vLLM

- Vulnerability Management

- Basic Model

- Registering a Model

- Unit Testing

- Multi-Modal Support

- Speech-to-Text (Transcription/Translation) Support

- CI CI

- CI Failures

- Nightly Builds of vLLM Wheels

- Update PyTorch version on vLLM OSS CI/CD

- Design Documents Design Documents

- Plugins Plugins

- Endpoint Plugins

- IO Processor Plugins

- LoRA Resolver Plugins

- Plugin System

- Architecture Overview Architecture Overview Table of contents

- Entrypoints

- LLM Class

- Online Serving

- V1 Process Architecture

- API Server Process

- Engine Core Process

- GPU Worker Processes

- DP Coordinator Process (conditional)

- Process Count Summary

- LLM Engine

- LLMEngine

- AsyncLLMEngine

- Worker

- Model Runner

- Model

- Class Hierarchy

- Attention Backend Feature Support

- CUDA Graphs

- Vision Encoder (ViT) CUDA Graphs

- CustomOp

- Dual Batch Overlap

- How to debug the vLLM-torch.compile integration

- Fused MoE Modular Kernel

- Fusion torch.compile passes

- Integration with Hugging Face

- Hybrid KV Cache Manager

- Logits Processors

- Metrics

- Multi-Modal Data Processing

- Model Runner V2 Design Document

- Fused MoE Kernel Features

- Python Multiprocessing

- NIXL KV Cache Lease Renewal

- NIXL push-mode KV transfer

- Optimization Levels

- Paged Attention

- Automatic Prefix Caching

- torch.compile integration

- torch.compile with Multimodal Encoders

- vLLM IR: Functional Intermediate Representation

- Benchmark CLI

- Parameter Sweeps

- Performance Dashboard

- collect_env

- connections

- env_override

- envs

- exceptions

- forward_context

- logger

- logits_process

- logprobs

- model_inspection

- outputs

- pooling_params

- sampling_params

- scalar_type

- scripts

- sequence

- tasks

- version

- audio

- base

- image

- video

- latency

- mm_processor

- plot

- serve

- startup

- throughput

- create_txt_slices_dataset

- datasets

- utils

- endpoint_request_func

- ready_checker

- utils

- cli

- param_sweep

- plot

- plot_pareto

- serve

- serve_workload

- server

- startup

- utils

- backends

- base_static_graph

- breakable_cudagraph

- caching

- codegen

- compiler_interface

- counter

- cuda_graph

- decorators

- monitor

- partition_rules

- piecewise_backend

- wrapper

- fx_utils

- inductor_pass

- pass_manager

- vllm_inductor_pass

- act_quant_fusion

- allreduce_rms_fusion

- attn_quant_fusion

- collective_fusion

- matcher_utils

- mla_attn_quant_fusion

- mla_rope_kvcache_cat_fusion

- qk_norm_rope_fusion

- qk_norm_rope_kvcache_fusion

- rms_quant_fusion

- rocm_aiter_fusion

- rope_kvcache_fusion

- sequence_parallelism

- clone_elimination

- inplace_functionalization

- lowering_pass

- utils

- fix_functionalization

- noop_elimination

- post_cleanup

- scatter_split_replace

- split_coalescing

- attention

- cache

- compilation

- device

- diffusion

- ec_transfer

- kernel

- kv_events

- kv_transfer

- load

- lora

- mamba

- model

- model_arch

- multimodal

- observability

- offload

- parallel

- pooler

- profiler

- quantization

- reasoning

- scheduler

- speculative

- speech_to_text

- structured_outputs

- utils

- vllm

- weight_transfer

- cvt

- cumem

- sleep_mode_backend

- xpumem

- communication_op

- kv_events

- nixl_utils

- parallel_state

- stateless_coordinator

- utils

- aiter_custom_all_reduce

- all2all

- all_reduce_utils

- base_device_communicator

- cpu_communicator

- cuda_communicator

- cuda_wrapper

- custom_all_reduce

- flashinfer_all_reduce

- mnnvl_compat

- pynccl

- pynccl_allocator

- pynccl_wrapper

- quick_all_reduce

- ray_communicator

- shm_broadcast

- shm_object_storage

- symm_mem

- xpu_communicator

- ec_transfer_state

- base

- example_connector

- factory

- common

- connector

- ec_shared_region

- embedding_cache

- step_tracker

- descriptor_buffers

- elastic_execute

- elastic_state

- standby_state

- async_worker

- eplb_communicator

- eplb_state

- eplb_utils

- rebalance_execute

- abstract

- default

- kv_transfer_state

- base

- factory

- utils

- base

- decode_bench_connector

- example_connector

- example_hidden_states_connector

- flexkv_connector

- lmcache_connector

- lmcache_mp_connector

- metrics

- multi_connector

- offloading_connector

- simple_cpu_offload_connector

- ssm_conv_transfer_utils

- hf3fs_client

- hf3fs_connector

- hf3fs_metadata_server

- common

- gather_scatter_helper

- hf3fs_mock_client

- multi_process_adapter

- utils

- vllm_v1_adapter

- mooncake_connector

- mooncake_utils

- rdma_utils

- stats

- connector

- coordinator

- data

- metrics

- protocol

- scheduler

- worker

- moriio_common

- moriio_connector

- moriio_engine

- moriio_layout

- base_scheduler

- base_worker

- connector

- metadata

- pull_scheduler

- pull_worker

- push_scheduler

- push_worker

- scheduler

- stats

- tp_mapping

- utils

- worker

- common

- config

- events

- metrics

- scheduler

- worker

- base

- clients

- factory

- ipc_engine

- nccl_common

- nccl_engine

- packed_tensor

- sparse_nccl_engine

- arg_utils

- async_llm_engine

- llm_engine

- protocol

- chat_utils

- grpc_server

- launcher

- llm

- offline_utils

- api_router

- protocol

- serving

- collect_env

- launch

- main

- openai

- run_batch

- serve

- types

- base

- latency

- main

- mm_processor

- serve

- startup

- sweep

- throughput

- api_router

- factories

- serving

- offline

- online

- utils

- api_router

- serving

- tool

- tool_server

- api_server

- cli_args

- dp_supervisor

- run_batch

- api_router

- batch_serving

- protocol

- serving

- api_router

- protocol

- serving

- protocol

- api_router

- protocol

- serving

- harmony_utils

- api_router

- context

- harmony

- protocol

- serving

- streaming_events

- utils

- factories

- offline

- typing

- utils

- io_processor

- protocol

- serving

- api_router

- io_processor

- protocol

- serving

- api_router

- io_processor

- protocol

- serving

- api_router

- io_processor

- protocol

- serving

- api_router

- io_processor

- protocol

- serving

- typing

- utils

- factories

- api_router

- serving

- api_router

- serving

- api_router

- mm_serde

- protocol

- serving

- api_router

- middleware

- serving

- typing

- basic

- health

- metrics

- offline_docs

- api_router

- protocol

- api_router

- protocol

- serving

- api_utils

- constants

- error_response

- fingerprint

- orca_metrics

- request_logger

- server_utils

- ssl

- tool_calls_utils

- factories

- protocol

- serving

- utils

- api_router

- connection

- metrics

- protocol

- serving

- api_router

- protocol

- serving

- api_router

- protocol

- serving

- engine

- llm

- preprocess

- op

- tolerances

- util

- layernorm

- aiter_ops

- oink_ops

- vllm_c

- xpu_ops

- case_key

- config_manager

- register

- utils

- dynamic_per_token_scaled_fp8_quant

- fused_qk_norm_rope

- per_token_group_fp8_quant

- rms_norm_dynamic_per_token_quant

- rms_norm_per_block_quant

- silu_and_mul_per_block_quant

- silu_mul_fp8

- qkv_padded_fp8_quant

- access_log_filter

- dump_input

- formatter

- lazy

- log_time

- torch_tensor

- lora_model

- lora_weights

- model_manager

- peft_helper

- request

- resolver

- utils

- worker_manager

- base

- base_linear

- column_parallel_linear

- fused_moe

- logits_processor

- replicated_linear

- row_parallel_linear

- utils

- vocal_parallel_embedding

- lora_ops

- fp8_kernel_utils

- fused_moe_lora_fp8_op

- fused_moe_lora_op

- kernel_utils

- lora_expand_fp8_op

- lora_expand_op

- lora_kernel_metadata

- lora_shrink_fp8_op

- lora_shrink_op

- utils

- lora_ops

- punica_base

- punica_cpu

- punica_gpu

- punica_selector

- punica_xpu

- utils

- custom_op

- parameter

- utils

- dcp_indexer_cutedsl

- base

- zentorch_utils

- ll_bf16

- allspark

- conch

- cpu

- cutlass

- dynamic_4bit

- exllama

- humming

- MPLinearKernel

- machete

- marlin

- rdna3_w4a16

- rdna_hybrid_w4a16

- triton_w4a16

- xpu

- zentorch

- base

- flashinfer

- humming

- marlin

- xpu

- emulation

- flashinfer

- humming

- Mxfp8LinearKernel

- marlin

- rocm_native

- xpu

- base

- cutlass

- emulation

- fbgemm

- flashinfer

- humming

- marlin

- aiter

- BlockScaledMMLinearKernel

- cpu

- cutlass

- deep_gemm

- flashinfer

- humming

- marlin

- pytorch

- rocm

- ScaledMMLinearKernel

- triton

- xpu

- zentorch

- aiter

- tilelang

- tilelang_kernels

- torch

- triton

- activation

- attention_layer_base

- batch_invariant

- conv

- fused_allreduce_gemma_rms_norm

- fused_qk_norm_rope

- layernorm

- lightning_attn

- linear

- logits_processor

- mhc

- mla

- resampler

- sparse_attn_indexer

- utils

- vocab_parallel_embedding

- attention

- chunked_local_attention

- cross_attention

- encoder_only_attention

- kv_transfer_utils

- mla_attention

- mm_encoder_attention

- pcp

- prefill_prefix_lm_attention

- rswa_attention

- sparse_mla_attention

- static_sink_attention

- activation

- all2all_utils

- config

- cpu_fused_moe

- deep_gemm_utils

- eep_reconfigure

- expert_map_manager

- fused_flydsl_moe

- fused_moe

- fused_moe_method_base

- fused_moe_modular_method

- hpc_moe

- layer

- modular_kernel

- moe_align_block_size

- moe_fused_mul_sum

- moe_permute_unpermute

- routed_experts

- routed_experts_capturer

- topk_weight_and_reduce

- unquantized_fused_moe_method

- utils

- aiter_mxfp4_w4a8_moe

- aiter_mxfp8_moe

- batched_deep_gemm_moe

- cpu_int4_moe

- cpu_moe

- cutlass_moe

- deep_gemm_moe

- fallback

- flashinfer_b12x_moe

- flashinfer_cutedsl_batched_moe

- flashinfer_cutedsl_moe

- flashinfer_cutlass_moe

- fused_batched_moe

- fused_humming_moe

- gpt_oss_triton_kernels_moe

- int4_emulation_moe

- lora_context

- lora_experts_mixin

- marlin_moe

- mxfp8_emulation_moe

- mxfp8_native_moe

- nvfp4_emulation_moe

- ocp_mx_emulation_moe

- rocm_aiter_moe

- triton_cutlass_moe

- triton_deep_gemm_moe

- triton_moe

- trtllm_bf16_moe

- trtllm_fp8_moe

- trtllm_lora_moe

- trtllm_mxfp4_moe

- trtllm_mxint4_moe

- trtllm_nvfp4_moe

- xpu_moe

- base

- fp8

- int8

- int_wna16

- mxfp4

- mxfp8

- nvfp4

- unquantized

- w4a8

- w4a8_int8

- batched

- deepep_ht

- deepep_ll

- deepep_v2

- flashinfer_nvlink_one_sided

- flashinfer_nvlink_two_sided

- mori

- naive_dp_ep

- nixl_ep

- no_dp_ep

- aiter_shared_routed_fused_moe_router

- base_router

- bf16x3_router_gemm_cutedsl

- custom_routing_router

- dsv4_topk

- fused_moe_router

- fused_topk_bias_router

- fused_topk_router

- gate_linear

- grouped_topk_router

- router_factory

- routing_simulator_router

- zero_expert_router

- moe_runner

- moe_runner_interface

- shared_experts

- quant_activation

- hpc_module

- rope_norm

- abstract

- mamba_mixer

- mamba_mixer2

- mamba_utils

- short_conv

- base

- kimi_gdn_linear_attn

- olmo_gdn_linear_attn

- qwen_gdn_linear_attn

- bailing_linear_attn

- base

- minimax_linear_attn

- causal_conv1d

- layernorm_gated

- mamba_ssm

- ssd_bmm

- ssd_chunk_scan

- ssd_chunk_state

- ssd_combined

- ssd_state_passing

- ssu_dispatch

- triton_helpers

- causal_conv1d

- gdn_attention

- kernel_h

- kernel_kkt_inv_uw

- kernel_o

- lamport_workspace

- rms_norm_tp

- abstract

- activations

- common

- special

- heads

- methods

- poolers

- heads

- methods

- poolers

- auto_awq

- auto_gptq

- awq_triton

- base_config

- bitsandbytes

- experts_int8

- fbgemm_fp8

- fp8

- fp_quant

- humming

- input_quant_fp8

- kv_cache

- modelopt

- moe_wna16

- mxfp4

- qutlass_utils

- torchao

- compressed_tensors

- compressed_tensors_embedding

- triton_scaled_mm

- utils

- compressed_tensors_moe

- compressed_tensors_moe_w4a4_mxfp4

- compressed_tensors_moe_w4a4_nvfp4

- compressed_tensors_moe_w4a8_fp8

- compressed_tensors_moe_w4a8_int8

- compressed_tensors_moe_w4a16_flydsl

- compressed_tensors_moe_w8a8_fp8

- compressed_tensors_moe_w8a8_int8

- compressed_tensors_moe_w8a8_mxfp8

- compressed_tensors_moe_wna16

- compressed_tensors_moe_wna16_marlin

- compressed_tensors_moe_wna16_rdna3

- rocm_moe_rdna

- compressed_tensors_scheme

- compressed_tensors_w4a4_mxfp4

- compressed_tensors_w4a4_nvfp4

- compressed_tensors_w4a8_fp8

- compressed_tensors_w4a8_int

- compressed_tensors_w8a8_fp8

- compressed_tensors_w8a8_int8

- compressed_tensors_w8a8_mxfp8

- compressed_tensors_w8a16_fp8

- compressed_tensors_wNa4

- compressed_tensors_wNa8

- compressed_tensors_wNa8o8

- compressed_tensors_wNa16

- linear

- module

- utils

- linear_qutlass_nvfp4

- config_parser

- inc

- inc_linear

- factory

- inc_ark_ops

- inc_scheme

- inc_wna16_linear

- inc_wna16_scheme

- base

- fp8

- int8

- moe_base

- mxfp8

- nvfp4

- quark

- quark_moe

- utils

- quark_nvfp4

- quark_ocp_mx

- quark_scheme

- quark_w4a8_mxfp4_fp8

- quark_w8a8_fp8

- quark_w8a8_int8

- centroids

- config

- allspark_utils

- flashinfer_fp4_moe

- flashinfer_mxint4_moe

- flashinfer_utils

- fp8_utils

- gptq_utils

- humming_utils

- int8_utils

- layer_utils

- machete_utils

- marlin_utils

- marlin_utils_fp4

- marlin_utils_fp8

- marlin_utils_test

- mxfp4_utils

- mxfp6_utils

- mxfp8_utils

- nvfp4_emulation_utils

- nvfp4_utils

- ocp_mx_utils

- quant_utils

- w8a8_utils

- base

- common

- deepseek_scaling_rope

- dual_chunk_rope

- dynamic_ntk_alpha_rope

- dynamic_ntk_scaling_rope

- ernie45_vl_rope

- fope

- gemma4_rope

- linear_scaling_rope

- llama3_rope

- llama4_vision_rope

- mrope

- mrope_interleaved

- ntk_scaling_rope

- phi3_long_rope_scaled_rope

- telechat3_scaling_rope

- xdrope

- yarn_scaling_rope

- base_loader

- bitsandbytes_loader

- default_loader

- dummy_loader

- ep_weight_filter

- modelexpress_loader

- runai_streamer_loader

- sharded_state_loader

- tensorizer

- tensorizer_loader

- utils

- weight_utils

- layerwise

- meta

- sanitize

- torchao_decorator

- types

- utils

- AXK1

- adapters

- afmoe

- aimv2

- apertus

- arcee

- arctic

- aria

- audioflamingo3

- bagel

- bailing_moe

- bailing_moe_linear

- bailing_moe_mtp

- bee

- bert

- bert_with_rope

- blip

- blip2

- bloom

- chameleon

- chatglm

- cheers

- clip

- cohere2_moe

- cohere2_vision

- cohere_asr

- cohere_eagle

- colbert

- colmodernvbert

- colpali

- colqwen3

- colqwen3_5

- commandr

- config

- conformer_encoder

- cosmos3

- cosmos3_edge

- dbrx

- deepencoder

- deepencoder2

- deepseek_eagle

- deepseek_eagle3

- deepseek_mtp

- deepseek_ocr

- deepseek_ocr2

- deepseek_v2

- deepseek_vl2

- diffusion_gemma

- dots_ocr

- eagle2_5_vl

- ernie45

- ernie45_moe

- ernie45_vl

- ernie45_vl_moe

- ernie_mtp

- exaone

- exaone4

- exaone4_5

- exaone4_5_mtp

- exaone_moe

- exaone_moe_mtp

- extract_hidden_states

- fairseq2_llama

- falcon

- falcon_h1

- fireredasr2

- fireredlid

- flex_olmo

- funasr

- funaudiochat

- gemma

- gemma2

- gemma3

- gemma3_mm

- gemma3n

- gemma3n_audio_utils

- gemma3n_mm

- gemma4

- gemma4_dspark

- gemma4_mm

- gemma4_mtp

- gemma4_unified

- glm

- glm4

- glm4_1v

- glm4_moe

- glm4_moe_lite

- glm4_moe_lite_mtp

- glm4_moe_mtp

- glm4v

- glm_ocr

- glm_ocr_mtp

- glmasr

- glmasr_utils

- gpt2

- gpt_j

- gpt_neox

- gpt_oss

- granite

- granite4_vision

- granite_speech

- granite_speech_plus

- granitemoe

- granitemoehybrid

- granitemoeshared

- gritlm

- h2ovl

- hrm_text

- hunyuan_v1

- hunyuan_vision

- hy_v3

- hy_v3_mtp

- hyperclovax

- hyperclovax_vision

- hyperclovax_vision_v2

- idefics2_vision_model

- idefics3

- interfaces

- interfaces_base

- intern_vit

- internlm2

- interns1

- interns1_pro

- interns1_vit

- interns2_preview

- internvl

- iquest_loopcoder

- isaac

- jais2

- jamba

- jina

- jina_vl

- kanana_v

- keye

- keye_vl1_5

- kimi_audio

- kimi_k25

- kimi_k25_vit

- kimi_linear

- kimi_vl

- laguna

- laguna_dflash

- lfm2

- lfm2_moe

- lfm2_siglip2

- lfm2_vl

- lightonocr

- llama

- llama4

- llama4_eagle

- llama_eagle

- llama_eagle3

- llava

- llava_next

- llava_next_video

- llava_onevision

- llava_onevision2

- longcat_flash

- longcat_flash_mtp

- longcat_flash_ngram

- mamba

- mamba2

- medusa

- mellum

- midashenglm

- mimo

- mimo_audio

- mimo_mtp

- mimo_v2

- mimo_v2_mtp

- mimo_v2_omni

- minicpm

- minicpm3

- minicpm_eagle

- minicpmo

- minicpmv

- minicpmv4_6

- minimax_m2

- mistral

- mistral3

- mistral_eagle

- mistral_large_3

- mistral_large_3_eagle

- mixtral

- mllama4

- mlp_speculator

- modernbert

- module_mapping

- molmo

- molmo2

- moondream3

- moonvit

- moss_audio

- moss_transcribe_diarize

- mpt

- nano_nemotron_vl

- nemotron

- nemotron_h

- nemotron_h_mtp

- nemotron_nas

- nemotron_parse

- nemotron_vl

- nvlm_d

- olmo3

- olmo_hybrid

- olmoe

- openai_privacy_filter

- opencua

- openpangu

- openpangu_mtp

- openpangu_vl

- openvla

- opt

- orion

- ouro

- ovis

- ovis2_5

- paddleocr_vl

- paligemma

- parakeet

- param2moe

- phi

- phi3

- phi3v

- phi4mm

- phi4mm_audio

- phi4mm_utils

- phi4siglip

- phimoe

- pixtral

- plamo2

- plamo3

- qianfan_ocr

- qwen2

- qwen2_5_omni_thinker

- qwen2_5_vl

- qwen2_audio

- qwen2_moe

- qwen2_rm

- qwen2_vl

- qwen3

- qwen3_5

- qwen3_5_mtp

- qwen3_asr

- qwen3_asr_forced_aligner

- qwen3_asr_realtime

- qwen3_dflash

- qwen3_dspark

- qwen3_eagle3

- qwen3_moe

- qwen3_next

- qwen3_next_mtp

- qwen3_omni_moe_thinker

- qwen3_vl

- qwen3_vl_moe

- radio

- registry

- rnj1

- roberta

- rvl

- sarvam

- seed_oss

- siglip

- siglip2navit

- skyworkr1v

- smolvlm

- solar

- stablelm

- step1

- step3_text

- step3_vl

- step3p5

- step3p5_mtp

- step3p7

- step_vl

- telechat2

- teleflm

- terratorch

- ultravox

- unlimited_ocr

- utils

- vision

- voxtral

- voxtral_realtime

- voyage

- whisper

- whisper_causal

- whisper_utils

- zamba2

- base

- causal

- fuser

- fx_utils

- legacy

- moe

- multimodal

- pooling

- utils

- base

- glu

- moe

- qkv

- rms_norm

- base

- prefetch

- prefetch_ops

- uva

- cutedsl_warmup

- deep_gemm_warmup

- deepseek_v4_mhc_warmup

- fa4_cutedsl_config

- flashinfer_autotune_cache

- flashinfer_sparse_mla_warmup

- kernel_warmup

- minimax_m3_msa_warmup

- qwen_triton_warmup

- sparse_mla_triton_warmup

- v1_block_table_warmup

- attention

- compressor

- quant_config

- sparse_mla

- dspark

- model

- mtp

- rocm

- rope

- cache_utils

- fused_compress_quant_cache

- fused_indexer_q

- fused_inv_rope_fp8_quant

- fused_mtp_input_rmsnorm

- fused_qk_rmsnorm

- save_partial_states

- dspark

- flashinfer_sparse

- flashmla

- model

- mtp

- dequant_gather_k_cutedsl

- fused_indexer_q_cutedsl

- o_proj

- prepare_megamoe

- sparse_attn_compress_cutedsl

- dspark

- model

- mtp

- xpu_qnorm_rope_kv_fp8_insert

- xpu_sparse

- xpu_sparse_decode_fp8

- attention

- fused_ops

- kernels

- model

- mtp

- configs

- mm_preprocess

- towers

- attention

- layernorm

- logits_processor

- mlp

- model

- moe

- mtp

- sconv_swa_attn

- short_conv

- fa4_rel_attention

- fa4_warmup

- lamport

- mm_towers

- norm

- qkvr_prep

- sconv

- silu_and_mul

- model

- mtp

- sparse_attention_msa

- gemma_rmsnorm

- index_topk

- sparse_attn

- sparse_pa

- swiglu_oai

- indexer

- mm_preprocess

- sparse_attention

- vision_tower

- index_topk

- sparse_attn

- indexer_msa

- model

- mtp

- sparse_attention_msa

- index_decode_score

- audio

- cache

- encoder_budget

- evs

- gpu_ipc_memory

- hasher

- image

- inputs

- parse

- registry

- utils

- video

- audio

- base

- connector

- image

- video

- context

- dummy_inputs

- inputs

- processor

- abstract_parser

- deepseek_v4

- deepseek_v32

- gemma4

- glm47_moe

- harmony

- inkling

- kimi_k2

- metrics

- minimax_m2

- mistral

- nemotron_v3

- parser_manager

- qwen3

- seed_oss

- utils

- adapters

- events

- incremental_lexer

- parser_engine

- parser_engine_config

- registered_adapters

- streaming_parser_engine

- token_id_scanner

- cpu

- cuda

- interface

- rocm

- tpu

- xpu

- zen_cpu

- interface

- filesystem_resolver

- hf_hub_resolver

- layerwise_profile

- utils

- wrapper

- lazy_utils

- ray_env

- abs_reasoning_parsers

- basic_parsers

- cohere_command_reasoning_parser

- deepseek_r1_reasoning_parser

- deepseek_v3_reasoning_parser

- deepseek_v4_engine_reasoning_parser

- ernie45_reasoning_parser

- gemma4_engine_reasoning_parser

- gemma4_utils

- glm47_moe_reasoning_parser

- gptoss_reasoning_parser

- granite_reasoning_parser

- hunyuan_a13b_reasoning_parser

- hy_v3_reasoning_parser

- identity_reasoning_parser

- inkling_reasoning_parser

- kimi_k2_reasoning_parser

- minimax_m2_reasoning_parser

- minimax_m3_reasoning_parser

- mistral_reasoning_parser

- nemotron_v3_engine_reasoning_parser

- olmo3_reasoning_parser

- poolside_v1_reasoning_parser

- qwen3_engine_reasoning_parser

- seed_oss_engine_reasoning_parser

- step3_reasoning_parser

- step3p5_reasoning_parser

- base

- deepseek_v4

- deepseek_v32

- embed_utils

- hf

- inkling

- inkling_encoding

- mistral

- online_derenderer

- online_renderer

- params

- registry

- terratorch

- preprocess

- tokenize

- deepseek_v4

- deepseek_v4_encoding

- deepseek_v32

- deepseek_v32_encoding

- detokenizer_utils

- fastokens

- hf

- kimi_audio

- mistral

- protocol

- registry

- abstract_tool_parser

- apertus_tool_parser

- cohere_command_tool_parser

- deepseekv3_tool_parser

- deepseekv4_engine_tool_parser

- deepseekv31_tool_parser

- deepseekv32_engine_tool_parser

- ernie45_tool_parser

- functiongemma_tool_parser

- gemma4_engine_tool_parser

- gemma4_utils

- gigachat3_tool_parser

- glm47_moe_tool_parser

- gptoss_tool_parser

- granite4_tool_parser

- granite_20b_fc_tool_parser

- granite_tool_parser

- hermes_tool_parser

- hunyuan_a13b_tool_parser

- hy_v3_tool_parser

- inkling_tool_parser

- internlm2_tool_parser

- jamba_tool_parser

- kimi_k2_tool_parser

- lfm2_tool_parser

- llama4_pythonic_tool_parser

- llama_tool_parser

- longcat_tool_parser

- minicpm5xml_tool_parser

- minimax_m2_tool_parser

- minimax_m3_tool_parser

- mistral_tool_parser

- olmo3_tool_parser

- phi4mini_tool_parser

- poolside_v1_tool_parser

- pythonic_tool_parser

- qwen3_engine_tool_parser

- rust_tool_parser

- seed_oss_engine_tool_parser

- step3_tool_parser

- step3p5_tool_parser

- streaming

- structural_tag_registry

- utils

- xlam_tool_parser

- otel

- utils

- config

- config_parser_base

- dynamic_module

- model_arch_config_convertor

- processor

- repo_utils

- runai_utils

- s3_utils

- utils

- registry

- allocation

- force_first_config

- importing

- usage_lib

- argparse_utils

- async_utils

- cache

- collection_utils

- counter

- cpu_resource_utils

- cpu_triton_utils

- deep_gemm

- flashinfer

- func_utils

- gc_utils

- gpu_sync_debug

- hashing

- hpc

- humming

- import_utils

- jit_monitor

- jsontree

- math_utils

- mem_constants

- mem_utils

- mistral

- multi_stream_utils

- nccl

- network_utils

- numa_utils

- nvtx_pytorch_hooks

- ompmultiprocessing

- platform_utils

- print_utils

- registry

- serial_utils

- sparse_utils

- system_utils

- tensor_schema

- torch_utils

- tqdm_utils

- cudagraph_dispatcher

- kv_cache_interface

- kv_cache_spec_registry

- outputs

- request

- serial_utils

- utils

- backend

- selector

- cpu_attn

- fa_utils

- flash_attn

- flash_attn_diffkv

- flashinfer

- flex_attention

- gdn_attn

- hpc_attn

- linear_attn

- mamba1_attn

- mamba2_attn

- mamba_attn

- registry

- rocm_aiter_fa

- rocm_aiter_unified_attn

- rocm_attn

- short_conv_attn

- triton_attn

- triton_attn_diffkv

- turboquant_attn

- utils

- aiter_triton_mla

- compressor_utils

- cutlass_mla

- flashattn_mla

- flashattn_mla_sparse

- flashinfer_mla

- flashinfer_mla_sparse

- flashinfer_mla_sparse_sm120

- flashmla

- flashmla_sparse

- indexer

- rocm_aiter_mla

- rocm_aiter_mla_sparse

- sparse_swa

- sparse_utils

- tokenspeed_mla

- triton_mla

- xpu_mla_sparse

- aiter_flash_attn

- base

- flash_attn

- flashinfer

- registry

- selector

- tokenspeed_mla

- trtllm_ragged

- chunked_prefill_paged_decode

- common

- dcp_alltoall

- flashmla

- int4_per_token_head

- merge_attn_states

- paged_attn

- prefix_prefill

- rocm_aiter_mla_sparse

- triton_attention_helpers

- triton_decode_attention

- triton_fp8_mqa_logits

- triton_merge_attn_states

- triton_prefill_attention

- triton_reshape_and_cache_flash

- triton_turboquant_decode

- triton_turboquant_store

- triton_unified_attention

- triton_unified_attention_diffkv

- vit_attn_wrappers

- xpu_mla_sparse

- block_pool

- encoder_cache_manager

- kv_cache_coordinator

- kv_cache_manager

- kv_cache_metrics

- kv_cache_utils

- single_type_kv_cache_manager

- async_scheduler

- interface

- output

- request_queue

- scheduler

- utils

- async_llm

- coordinator

- core

- core_client

- detokenizer

- exceptions

- input_processor

- llm_engine

- logprobs

- output_processor

- parallel_sampling

- tensor_ipc

- utils

- abstract

- multiproc_executor

- ray_env_utils

- ray_executor

- ray_executor_v2

- ray_utils

- uniproc_executor

- vllm_net_devices

- base

- config

- factory

- file_mapper

- common

- gpu_worker

- manager

- shared_offload_region

- spec

- swap_blocks_triton

- arc

- base

- lru

- async_lookup

- base

- factory

- manager

- spec

- manager

- io

- manager

- thread_pool

- config

- manager

- base

- zmq

- base

- nixl

- client

- protocol

- server

- session

- loggers

- perf

- prometheus

- ray_wrappers

- reader

- stats

- utils

- late_interaction

- late_interaction_runner

- metadata

- rejection_sampler

- sampler

- thinking_budget_state

- builtin

- interface

- state

- bad_words

- logprobs

- penalties

- topk_topp_sampler

- topk_topp_triton

- copy_backend

- cuda_mem_ops

- manager

- metadata

- worker

- custom_class_proposer

- dflash

- draft_model

- eagle

- extract_hidden_states

- gemma4

- llm_base_proposer

- medusa

- metadata

- metrics

- ngram_proposer

- ngram_proposer_gpu

- step3p5

- suffix_decoding

- utils

- vocab_mapping

- utils

- backend_guidance

- backend_lm_format_enforcer

- backend_outlines

- backend_types

- backend_xgrammar

- request

- utils

- block_table

- cp_utils

- cpu_model_runner

- cpu_worker

- dp_utils

- ec_connector_model_runner_mixin

- encoder_cudagraph

- encoder_cudagraph_defs

- gpu_input_batch

- gpu_model_runner

- gpu_ubatch_wrapper

- gpu_worker

- kv_connector_model_runner_mixin

- lora_model_runner_mixin

- mamba_utils

- startup_plan

- tpu_input_batch

- ubatch_utils

- ubatching

- utils

- worker_base

- workspace

- xpu_model_runner

- xpu_worker

- buffer_utils

- model_runner

- shm

- async_utils

- attn_utils

- block_table

- buffer_utils

- cp_utils

- cudagraph_utils

- dp_utils

- eplb_utils

- input_batch

- kv_connector

- lora_utils

- model_runner

- pcp_manager

- pp_utils

- shutdown

- states

- structured_outputs

- warmup

- logits

- encoder_cache

- encoder_runner

- lora

- rope

- default

- encoder_decoder

- interface

- mamba_hybrid

- mm_pruning

- pooling_runner

- bad_words

- gumbel

- logit_bias

- logprob

- min_p

- output

- penalties

- prompt_logprob

- sampler

- states

- rejection_sampler

- rejection_sampler_utils

- speculator

- utils

- cudagraph_utils

- speculator

- cudagraph

- speculator

- utils

- speculator

- utils

- eagle3_utils

- speculator

- utils

- speculator

- vllm serve

- vllm chat

- vllm complete

- vllm run-batch

- vllm bench vllm bench

- vllm bench latency

- vllm bench mm-processor

- vllm bench serve

- vllm bench sweep plot

- vllm bench sweep plot_pareto

- vllm bench sweep serve

- vllm bench sweep serve_workload

- vllm bench throughput

- vllm launch vllm launch

- vllm launch render

- Community Community

- Contact Us

- Meetups

- Sponsors

- Governance Governance

- Collaboration Policy

- Committers

- Governance Process

- Blog

- Forum

- Slack

- Entrypoints

- LLM Class

- Online Serving

- V1 Process Architecture

- API Server Process

- Engine Core Process

- GPU Worker Processes

- DP Coordinator Process (conditional)

- Process Count Summary

- LLM Engine

- LLMEngine

- AsyncLLMEngine

- Worker

- Model Runner

- Model

- Class Hierarchy

- Home

- Developer Guide

- Design Documents

# Architecture Overview¶

This document provides an overview of the vLLM architecture.

- Architecture Overview

- Entrypoints

- LLM Class

- Online Serving

- V1 Process Architecture

- API Server Process

- Engine Core Process

- GPU Worker Processes

- DP Coordinator Process (conditional)

- Process Count Summary

- LLM Engine

- LLMEngine

- AsyncLLMEngine

- Worker

- Model Runner

- Model

- Class Hierarchy

## Entrypoints¶

vLLM provides a number of entrypoints for interacting with the system. The following diagram shows the relationship between them.

### LLM Class¶

The LLM class provides the primary Python interface for doing offline inference, which is interacting with a model without using a separate model inference server.

Here is a sample of LLM class usage:

```
from vllm import LLM, SamplingParams

# Define a list of input prompts
prompts = [
 "Hello, my name is",
 "The capital of France is",
 "The largest ocean is",
]

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize the LLM engine with the OPT-125M model
llm = LLM(model="facebook/opt-125m")

# Generate outputs for the input prompts
outputs = llm.generate(prompts, sampling_params)

# Print the generated outputs
for output in outputs:
 prompt = output.prompt
 generated_text = output.outputs[0].text
 print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

More API details can be found in the Offline Inference section of the API docs.

The code for the LLM class can be found in vllm/entrypoints/llm.py.

### Online Serving¶

The second primary interface to vLLM is via its online server. This server can be started using the vllm serve command.

```
vllm serve <model>
```

The code for the vllm CLI can be found in vllm/entrypoints/cli/main.py.

Sometimes you may see the API server entrypoint used directly instead of via the vllm CLI command. For example:

```
python -m vllm.entrypoints.openai.api_server --model <model>
```

Warning

python -m vllm.entrypoints.openai.api_server is deprecated and may become unsupported in a future release.

That code can be found in vllm/entrypoints/openai/api_server.py.

More details on the API server can be found in the Online Serving document.

## V1 Process Architecture¶

vLLM V1 uses a multi-process architecture to separate concerns and maximize throughput. Understanding this architecture is important for properly sizing CPU resources in your deployment. The key processes are:

### API Server Process¶

The API server process handles HTTP requests (e.g., the OpenAI-compatible API), performs input processing (tokenization, multi-modal data loading), and streams results back to clients. It communicates with the engine core process(es) via ZMQ sockets.

By default, there is 1 API server process, but when data parallelism is used, the API server count automatically scales to match the data parallel size. This can also be manually configured with the --api-server-count flag. Each API server connects to all engine cores via ZMQ in a many-to-many topology, enabling any API server to route requests to any engine core. Each API server process uses multiple CPU threads for media loading (controlled by VLLM_MEDIA_LOADING_THREAD_COUNT, default 8).

The code can be found in vllm/entrypoints/openai/api_server.py and vllm/v1/utils.py.

### Engine Core Process¶

The engine core process runs the scheduler, manages KV cache, and coordinates model execution across GPU workers. It runs a busy loop that continuously schedules requests and dispatches work to the GPU workers.

There is 1 engine core process per data parallel rank. For example, with --data-parallel-size 4, there are 4 engine core processes.

The code can be found in vllm/v1/engine/core.py and vllm/v1/engine/utils.py.

### GPU Worker Processes¶

Each GPU is managed by a dedicated worker process. The worker process loads model weights, executes forward passes, and manages GPU memory. Workers communicate with the engine core process that owns them.

There is 1 worker process per GPU. The total number of GPU worker processes equals tensor_parallel_size x pipeline_parallel_size per engine core.

The code can be found in vllm/v1/executor/multiproc_executor.py and vllm/v1/worker/gpu_worker.py.

### DP Coordinator Process (conditional)¶

When using data parallelism (--data-parallel-size > 1), an additional coordinator process manages load balancing across DP ranks and coordinates synchronized forward passes for MoE models.

There is 1 DP coordinator process (only when data parallelism is enabled).

The code can be found in vllm/v1/engine/coordinator.py.

### Process Count Summary¶

For a deployment with N GPUs, TP tensor parallel size, DP data parallel size, and A API server count:

Process Type Count Notes

API Server A (default DP) Handles HTTP requests and input processing

Engine Core DP (default 1) Scheduler and KV cache management

GPU Worker N (= DP x PP x TP) One per GPU, executes model forward passes

DP Coordinator 1 if DP > 1, else 0 Load balancing across DP ranks

Total A + DP + N (+ 1 if DP > 1)

For example, a typical single-node deployment with 4 GPUs (vllm serve -tp=4) has:

- 1 API server + 1 engine core + 4 GPU workers = 6 processes

A data parallel deployment with 8 GPUs (vllm serve -tp=2 -dp=4) has:

- 4 API servers + 4 engine cores + 8 GPU workers + 1 DP coordinator = 17 processes

For CPU resource sizing recommendations, see CPU Resources for GPU Deployments.

## LLM Engine¶

The LLMEngine and AsyncLLMEngine classes are central to the functioning of the vLLM system, handling model inference and asynchronous request processing.

### LLMEngine¶

The LLMEngine class is the core component of the vLLM engine. It is responsible for receiving requests from clients and generating outputs from the model. The LLMEngine includes input processing, model execution (possibly distributed across multiple hosts and/or GPUs), scheduling, and output processing.

- Input Processing: Handles tokenization of input text using the specified tokenizer.

- Scheduling: Chooses which requests are processed in each step.

- Model Execution: Manages the execution of the language model, including distributed execution across multiple GPUs.

- Output Processing: Processes the outputs generated by the model, decoding the token IDs from a language model into human-readable text.

The code for LLMEngine can be found in vllm/engine/llm_engine.py.

### AsyncLLMEngine¶

The AsyncLLMEngine class is an asynchronous wrapper for the LLMEngine class. It uses asyncio to create a background loop that continuously processes incoming requests. The AsyncLLMEngine is designed for online serving, where it can handle multiple concurrent requests and stream outputs to clients.

The OpenAI-compatible API server uses the AsyncLLMEngine. There is also a demo API server that serves as a simpler example in examples/applications/api_server/server.py.

The code for AsyncLLMEngine can be found in vllm/engine/async_llm_engine.py.

## Worker¶

A worker is a process that runs the model inference. vLLM follows the common practice of using one process to control one accelerator device, such as GPUs. For example, if we use tensor parallelism of size 2 and pipeline parallelism of size 2, we will have 4 workers in total. Workers are identified by their rank and local_rank. rank is used for global orchestration, while local_rank is mainly used for assigning the accelerator device and accessing local resources such as the file system and shared memory.

## Model Runner¶

Every worker has one model runner object, responsible for loading and running the model. Much of the model execution logic resides here, such as preparing input tensors and capturing cudagraphs.

## Model¶

Every model runner object has one model object, which is the actual torch.nn.Module instance. See huggingface_integration for how various configurations affect the class we ultimately get.

## Class Hierarchy¶

The following figure shows the class hierarchy of vLLM:

There are several important design choices behind this class hierarchy:

1. Extensibility: All classes in the hierarchy accept a configuration object containing all the necessary information. The VllmConfig class is the main configuration object that is passed around. The class hierarchy is quite deep, and every class needs to read the configuration it is interested in. By encapsulating all configurations in one object, we can easily pass the configuration object around and access the configuration we need. Suppose we want to add a new feature (this is often the case given how fast the field of LLM inference is evolving) that only touches the model runner. We will have to add a new configuration option in the VllmConfig class. Since we pass the whole config object around, we only need to add the configuration option to the VllmConfig class, and the model runner can access it directly. We don't need to change the constructor of the engine, worker, or model class to pass the new configuration option.

2. Uniformity: The model runner needs a unified interface to create and initialize the model. vLLM supports more than 50 types of popular open-source models. Each model has its own initialization logic. If the constructor signature varies with models, the model runner does not know how to call the constructor accordingly, without complicated and error-prone inspection logic. By making the constructor of the model class uniform, the model runner can easily create and initialize the model without knowing the specific model type. This is also useful for composing models. Vision-language models often consist of a vision model and a language model. By making the constructor uniform, we can easily create a vision model and a language model and compose them into a vision-language model.

Note

To support this change, all vLLM models' signatures have been updated to:

```
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

To avoid accidentally passing incorrect arguments, the constructor is now keyword-only. This ensures that the constructor will raise an error if old configurations are passed. vLLM developers have already made this change for all models within vLLM. For out-of-tree registered models, developers need to update their models, for example by adding shim code to adapt the old constructor signature to the new one:

```
class MyOldModel(nn.Module):
 def __init__(
 self,
 config,
 cache_config: Optional[CacheConfig] = None,
 quant_config: Optional[QuantizationConfig] = None,
 lora_config: Optional[LoRAConfig] = None,
 prefix: str = "",
 ) -> None:
 ...

from vllm.config import VllmConfig
class MyNewModel(MyOldModel):
 def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
 config = vllm_config.model_config.hf_config
 cache_config = vllm_config.cache_config
 quant_config = vllm_config.quant_config
 lora_config = vllm_config.lora_config
 super().__init__(config, cache_config, quant_config, lora_config, prefix)

from packaging import version
if version.parse(__version__) >= version.parse("0.6.4"):
 MyModel = MyNewModel
else:
 MyModel = MyOldModel
```

This way, the model can work with both old and new versions of vLLM.

3. Sharding and Quantization at Initialization: Certain features require changing the model weights. For example, tensor parallelism needs to shard the model weights, and quantization needs to quantize the model weights. There are two possible ways to implement this feature. One way is to change the model weights after the model is initialized. The other way is to change the model weights during the model initialization. vLLM chooses the latter. The first approach is not scalable to large models. Suppose we want to run a 405B model (with roughly 810GB weights) with 16 H100 80GB GPUs. Ideally, every GPU should only load 50GB weights. If we change the model weights after the model is initialized, we need to load the full 810GB weights to every GPU and then shard the weights, leading to a huge memory overhead. Instead, if we shard the weights during the model initialization, every layer will only create a shard of the weights it needs, leading to a much smaller memory overhead. The same idea applies to quantization. Note that we also add an additional argument prefix to the model's constructor so that the model can initialize itself differently based on the prefix. This is useful for non-uniform quantization, where different parts of the model are quantized differently. The prefix is usually an empty string for the top-level model and a string like "vision" or "language" for the sub-models. In general, it matches the name of the module's state dict in the checkpoint file.

One disadvantage of this design is that it is hard to write unit tests for individual components in vLLM because every component needs to be initialized by a complete config object. We solve this problem by providing a default initialization function that creates a default config object with all fields set to None. If the component we want to test only cares about a few fields in the config object, we can create a default config object and set the fields we care about. This way, we can test the component in isolation. Note that many tests in vLLM are end-to-end tests that test the whole system, so this is not a big problem.

In summary, the complete config object VllmConfig can be treated as an engine-level global state that is shared among all vLLM classes.

Made with Material for MkDocs
