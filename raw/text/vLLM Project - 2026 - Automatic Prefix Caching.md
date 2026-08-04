# vLLM Automatic Prefix Caching

- Source HTML: `raw/html/vLLM Project - 2026 - Automatic Prefix Caching.html`
- Source URL: https://docs.vllm.ai/en/stable/design/prefix_caching/
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

- Architecture Overview

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

- Automatic Prefix Caching Automatic Prefix Caching Table of contents

- Data Structure

- Operations

- Block Allocation

- Free

- Eviction (LRU)

- Example

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

- Data Structure

- Operations

- Block Allocation

- Free

- Eviction (LRU)

- Example

- Home

- Developer Guide

- Design Documents

# Automatic Prefix Caching¶

Prefix caching kv-cache blocks is a popular optimization in LLM inference to avoid redundant prompt computations. The core idea is simple – we cache the kv-cache blocks of processed requests, and reuse these blocks when a new request comes in with the same prefix as previous requests. Since prefix caching is almost a free lunch and won’t change model outputs, it has been widely used by many public endpoints (e.g., OpenAI, Anthropic, etc.) and most open source LLM inference frameworks (e.g., SGLang).

While there are many ways to implement prefix caching, vLLM chooses a hash-based approach. Specifically, we hash each kv-cache block by the tokens in the block and the tokens in the prefix before the block:

```
Block 1 Block 2 Block 3
 [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

In the example above, the KV cache in the first block can be uniquely identified with the token “A gentle breeze stirred”. The third block can be uniquely identified with the tokens in the block “laughed in the distance”, along with the prefix tokens “A gentle breeze stirred the leaves as children”. Therefore, we can build the block hash of hash(tuple[components]), where components are:

- Parent hash value: The hash value of the parent hash block.

- Block tokens: A tuple of tokens in this block. The reason to include the exact tokens is to reduce potential hash value collision.

- Extra hashes: Other values required to make this block unique, such as LoRA IDs, multi-modality input hashes (see the example below), and cache salts to isolate caches in multi-tenant environments.

Note 1

We only cache full blocks.

Note 2

In previous versions, the hash key was not guaranteed to be collision-free. As of v0.11, the default hashing algorithm is sha256, which addresses collision risks.

For vllm serve, you can control the hashing algorithm via --prefix-caching-hash-algo: - sha256 (default): Uses Python's pickle for serialization. Hashes may not be reproducible across different Python or vLLM versions. - sha256_cbor: Uses cbor2 for serialization, providing a reproducible, cross-language compatible hash. This is recommended for deterministic caching across environments. - xxhash: Uses Pickle serialization with xxHash (128-bit) for faster, non-cryptographic hashing. Requires the optional xxhash package. IMPORTANT: Use of a hashing algorithm that is not considered cryptographically secure theoretically increases the risk of hash collisions, which can cause undefined behavior or even leak private information in multi-tenant environments. Even if collisions are still very unlikely, it is important to consider your security risk tolerance against the performance benefits before turning this on. - xxhash_cbor combines canonical CBOR serialization with xxHash for reproducible hashing. Requires the optional xxhash package.

A hashing example with multi-modality inputs In this example, we illustrate how prefix caching works with multi-modality inputs (e.g., images). Assuming we have a request with the following messages:

```
messages = [
 {"role": "user",
 "content": [
 {"type": "text",
 "text": "What's in this image?"
 },
 {"type": "image_url",
 "image_url": {"url": image_url},
 },
 ]},
]
```

It will become the following prompt:

```
Prompt:
 <s>[INST]What's in this image?\n[IMG][/INST]

Tokenized prompt:
 [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 10, 4]

Prompt with placeholders (<P>):
 [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <P>, <P>, ..., <P>, 4]
```

As we can see, after the tokenization, the [IMG] will be replaced by a sequence of placeholder tokens, and these placeholders will be replaced by image embeddings during prefill. The challenge for prefix caching to support this case is we need to differentiate images from the placeholders. To address this problem, we encode the image hash generated by the frontend image processor. For example, the hash of the blocks in the above prompt would be (assuming block size 16, and we have 41 placeholder tokens):

```
Block 0
 Parent hash: None
 Token IDs: 1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <p>, ..., <p>
 Extra hash: <image hash>
Block 1
 Parent hash: Block 0 hash
 Token IDs: <p>, ..., <p>
 Extra hash: <image hash>
Block 2
 Parent hash: Block 1 hash
 Token IDs: <p>, ..., <p>
 Extra hash: <image hash>
Block 3
 Parent hash: Block 2 hash
 Token IDs: <p>, ..., <p>, 4
 Extra hash: <image hash>
```

In the rest of this document, we first introduce the data structure used for prefix caching in vLLM v1, followed by the prefix caching workflow of major KV cache operators (e.g., allocate, append, free, eviction). Finally, we use an example to illustrate the end to end prefix caching workflow.

Cache Isolation for Security To improve privacy in shared environments, vLLM supports isolating prefix cache reuse through optional per-request salting. By including a cache_salt in the request, this value is injected into the hash of the first block, ensuring that only requests with the same salt can reuse cached KV blocks. This prevents timing-based attacks where an adversary could infer cached content by observing latency differences. This offers protection without compromising performance.

```
{
 "messages": [
 {"role": "system", "content": "You are a helpful assistant."},
 {"role": "user", "content": "Here is a document with details about the world series: ..."},
 {"role": "user", "content": "Who won the world series in 2020?"}
 ],
 "cache_salt": "your-cache-salt"
}
```

With this setup, cache sharing is limited to users or requests that explicitly agree on a common salt, enabling cache reuse within a trust group while isolating others.

## Data Structure¶

The prefix caching in vLLM v1 is implemented in the KV cache manager. The basic building block is the “Block” data class (simplified):

```
class KVCacheBlock:
 # The block ID (immutable)
 block_id: int
 # The block hash (will be assigned when the block is full,
 # and will be reset when the block is evicted).
 block_hash: BlockHash
 # The number of requests using this block now.
 ref_cnt: int

 # The pointers to form a doubly linked list for the free queue.
 prev_free_block: "KVCacheBlock | None" = None
 next_free_block: "KVCacheBlock | None" = None
```

There are two design points to highlight:

- We allocate all KVCacheBlock when initializing the KV cache manager to be a block pool. This avoids Python object creation overheads and can easily track all blocks all the time.

- We introduce doubly linked list pointers directly in the KVCacheBlock, so that we could construct a free queue directly. This gives us two benefits:

- We could have O(1) complexity moving elements in the middle to the tail.

- We could avoid introducing another Python queue (e.g., deque) which has a wrapper to the elements.

As a result, we will have the following components when the KV cache manager is initialized:

- Block Pool: A list of KVCacheBlock.

- Free Block Queue: Only store the pointers of head and tail blocks for manipulations.

- Cache blocks: Mapping from hash key to block IDs.

- Request blocks: Mapping from request ID to allocated block IDs.

## Operations¶

### Block Allocation¶

New request: Workflow for the scheduler to schedule a new request with KV cache block allocation:

- The scheduler calls kv_cache_manager.get_computed_blocks() to get a sequence of blocks that have already been computed. This is done by hashing the prompt tokens in the request and looking up cache blocks.

- The scheduler calls kv_cache_manager.allocate_slots(). It does the following steps:

- Compute the number of new required blocks, and return if there are no sufficient blocks to allocate.

- “Touch” the computed blocks. It increases the reference count of the computed block by one, and removes the block from the free queue if the block wasn’t used by other requests. This is to avoid these computed blocks being evicted. See the example in the next section for illustration.

- Allocate new blocks by popping the heads of the free queue. If the head block is a cached block, this also “evicts” the block so that no other requests can reuse it anymore from now on.

- If an allocated block is already full of tokens, we immediately add it to the cache block, so that the block can be reused by other requests in the same batch.

Running request: Workflow for the scheduler to schedule a running request with KV cache block allocation:

- The scheduler calls kv_cache_manager.allocate_slots(). It does the following steps:

- Compute the number of new required blocks, and return if there are no sufficient blocks to allocate.

- Allocate new blocks by popping the heads of the free queue. If the head block is a cached block, this also “evicts” the block so that no other requests can reuse it anymore from now on.

- Append token IDs to the slots in existing blocks as well as the new blocks. If a block is full, we add it to the cache block to cache it.

Duplicated blocks Assuming block size is 4 and you send a request (Request 1) with prompt ABCDEF and decoding length 3:

```
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
 Tokens: [A, B, C, D, E, F, G]
 Block Table: [0 (ABCD), 1 (EFG)]
 Cache Blocks: 0
Time 1:
 Tokens: [A, B, C, D, E, F, G, H]
 Block Table: [0 (ABCD), 1 (EFGH)]
 Cache Blocks: 0, 1
Time 2:
 Tokens: [A, B, C, D, E, F, G, H, I]
 Block Table: [0 (ABCD), 1 (EFGH), 2 (I)]
 Cache Blocks: 0, 1
```

Now block 0 and block 1 are cached, and we send the same request again (Request 2) with greedy sampling, so that it will produce exactly the same outputs as the Request 1:

```
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
 Tokens: [A, B, C, D, E, F, G]
 Block Table: [0 (ABCD), 3 (EFG)]
 Cache Blocks: 0, 1
Time 1:
 Tokens: [A, B, C, D, E, F, G, H]
 Block Table: [0 (ABCD), 3 (EFGH)]
 Cache Blocks: 0, 1, 3
```

As can be seen, block 3 is a new full block and is cached. However, it is redundant as block 1, meaning that we cached the same block twice. In v0, when detecting block 3 is duplicated, we free block 3 and let Request 2 use block 1 instead, so its block table becomes [0, 1] in Time 1. However, the block table in vLLM v1 is append-only, meaning that changing the block table from [0, 3] to [0, 1] is not allowed. As a result, we will have duplicated blocks for the hash key E-H. This duplication will be eliminated when the request is freed.

### Free¶

When a request is finished, we free all its blocks if no other requests are using them (reference count = 0). In this example, we free request 1 and block 2, 3, 4, 8 associated with it. We can see that the freed blocks are added to the tail of the free queue in the reverse order. This is because the last block of a request must hash more tokens and is less likely to be reused by other requests. As a result, it should be evicted first.

### Eviction (LRU)¶

When the head block (least recently used block) of the free queue is cached, we have to evict the block to prevent it from being used by other requests. Specifically, eviction involves the following steps:

- Pop the block from the head of the free queue. This is the LRU block to be evicted.

- Remove the block ID from the cache block.

- Remove the block hash.

## Example¶

In this example, we assume the block size is 4 (each block can cache 4 tokens), and we have 10 blocks in the KV-cache manager in total.

Time 1: The cache is empty and a new request comes in. We allocate 4 blocks. 3 of them are already full and cached. The fourth block is partially full with 3 of 4 tokens.

Time 2: Request 0 makes the block 3 full and asks for a new block to keep decoding. We cache block 3 and allocate block 4.

Time 3: Request 1 comes in with the 14 prompt tokens, where the first 10 tokens are the same as request 0. We can see that only the first 2 blocks (8 tokens) hit the cache, because the 3rd block only matches 2 of 4 tokens.

Time 4: Request 0 is finished and free. Blocks 2, 3 and 4 are added to the free queue in the reverse order (but block 2 and 3 are still cached). Block 0 and 1 are not added to the free queue because they are being used by Request 1.

Time 5: Request 1 is finished and free.

Time 6: Request 2 comes in with the 29 prompt tokens, where the first 12 tokens are the same as request 0. Note that even the block order in the free queue was 7 - 8 - 9 - 4 - 3 - 2 - 6 - 5 - 1 - 0, the cache hit blocks (i.e., 0, 1, 2) are touched and removed from the queue before allocation, so the free queue becomes 7 - 8 - 9 - 4 - 3 - 6 - 5. As a result, the allocated blocks are 0 (cached), 1 (cached), 2 (cached), 7, 8, 9, 4, 3 (evicted).

Made with Material for MkDocs
