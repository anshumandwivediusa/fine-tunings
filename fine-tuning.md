# Fine-Tuning LLMs - Table of Content
## Module 1: Introduction to Fine-Tuning

### 1.1 What is Fine-Tuning?

* Definition
* Why fine-tune?
* Fine-tuning vs Prompt Engineering
* Fine-tuning vs RAG
* Fine-tuning vs In-Context Learning

### 1.2 When Should You Fine-Tune?

Decision Tree

```
Need latest knowledge?
        │
       Yes
        │
       RAG

Need new behavior?
        │
       Yes
        │
 Fine-Tuning

Need formatting only?
        │
       Prompt Engineering

Need company-specific knowledge?
        │
      RAG + Fine-Tuning
```

---

## Module 2: Foundation of Neural Networks

Before fine-tuning, understand how neural networks learn.

Topics

* Forward Propagation
* Backpropagation
* Gradient Descent
* Loss Functions
* Optimizers
* Learning Rate
* Epoch
* Batch Size
* Mini Batch Training
* Validation Set
* Test Set

---

## Module 3: Transformer Refresher

Review transformer architecture.

Topics

* Tokenization
* Embeddings
* Positional Encoding
* Multi Head Attention
* Self Attention
* Feed Forward Networks
* Layer Normalization
* Residual Connections
* Decoder-only Models
* Encoder Models
* Encoder-Decoder Models

---

## Module 4: Types of Fine-Tuning

### Full Fine-Tuning

Update every parameter.

Pros

* Highest accuracy

Cons

* Expensive

---

### Feature Extraction

Freeze model.

Train only classifier head.

---

### Parameter Efficient Fine-Tuning (PEFT)

Introduction

Why PEFT exists

Benefits

* Less GPU
* Faster
* Smaller checkpoints

---

## Module 5: LoRA (Most Important)

Topics

* What is LoRA
* Matrix decomposition
* Low Rank Approximation
* Rank (r)
* Alpha
* Dropout
* Target Modules

Math

```
Wnew = W + ΔW

ΔW = BA
```

Understand

* Why only A and B matrices train
* Memory savings
* GPU savings

---

## Module 6: Advanced PEFT Methods

Topics

* AdaLoRA
* IA3
* Prefix Tuning
* Prompt Tuning
* P-Tuning v2
* QLoRA
* DoRA
* VeRA

Comparison table

Memory

Speed

Accuracy

---

## Module 7: Quantization

Topics

Why Quantization

FP32

FP16

BF16

INT8

INT4

GPTQ

AWQ

BitsAndBytes

GGUF

Quantization aware training

Post Training Quantization

---

## Module 8: QLoRA

Topics

4-bit Training

NF4

Double Quantization

Paged Optimizer

Memory Optimization

When to use QLoRA

---

## Module 9: Dataset Preparation

Data Formats

Instruction Dataset

```
Instruction

Input

Output
```

Conversation Dataset

```
User

Assistant
```

ChatML

Alpaca

ShareGPT

OpenAI JSONL

Cleaning

Deduplication

Balancing

Data Augmentation

Synthetic Data Generation

---

## Module 10: Instruction Tuning

Topics

What is Instruction Tuning

Examples

General Instructions

Task Specific Instructions

Role Playing

Safety Alignment

Multi-turn Conversations

---

## Module 11: Supervised Fine-Tuning (SFT)

Topics

Training Objective

Cross Entropy Loss

Teacher Forcing

Label Masking

Packing

Sequence Length

Context Window

Checkpoint Saving

Resume Training

Evaluation

---

## Module 12: Preference Fine-Tuning

Topics

RLHF Overview

Human Feedback

Reward Model

Policy Model

Reference Model

---

### PPO

Understand

Policy Gradient

Reward

KL Penalty

Advantages

---

### DPO

Direct Preference Optimization

Why DPO replaced PPO

Chosen Response

Rejected Response

Loss Function

---

### ORPO

Odds Ratio Preference Optimization

---

### KTO

Kahneman-Tversky Optimization

---

## Module 13: Alignment

Topics

Harmlessness

Helpfulness

Honesty

Constitutional AI

AI Alignment

Safety Alignment

---

## Module 14: Fine-Tuning Frameworks

Learn

* Hugging Face Transformers
* TRL
* PEFT
* Accelerate
* DeepSpeed
* FSDP
* Unsloth
* Axolotl
* LlamaFactory

---

## Module 15: Hardware Optimization

Topics

CUDA

GPU Memory

VRAM

Gradient Checkpointing

Flash Attention

Mixed Precision

Gradient Accumulation

Distributed Training

Multi GPU

Tensor Parallelism

Pipeline Parallelism

Data Parallelism

ZeRO

---

## Module 16: Hyperparameter Tuning

Topics

Learning Rate

Scheduler

Warmup

Batch Size

Gradient Accumulation

Epoch

Weight Decay

Max Sequence Length

Packing

LoRA Rank

LoRA Alpha

LoRA Dropout

---

## Module 17: Evaluation

Automatic Metrics

BLEU

ROUGE

BERTScore

Perplexity

Exact Match

F1 Score

LLM-as-a-Judge

Human Evaluation

Win Rate

Pairwise Comparison

Hallucination Testing

Safety Testing

Bias Testing

---

## Module 18: Fine-Tuning Workflow

```
Raw Dataset

↓

Cleaning

↓

Formatting

↓

Tokenization

↓

Train / Validation Split

↓

Load Base Model

↓

Choose PEFT

↓

Train

↓

Save Adapter

↓

Merge Adapter

↓

Evaluate

↓

Deploy
```

---

## Module 19: Deployment

Topics

Merge LoRA

Export Model

GGUF

ONNX

TensorRT

vLLM

Text Generation Inference (TGI)

Ollama

Docker

Kubernetes

Inference Optimization

---

## Module 20: Production Best Practices

Topics

Experiment Tracking

MLflow

Weights & Biases

Model Registry

Versioning

Checkpoint Management

A/B Testing

Canary Deployment

Rollback

Monitoring

Latency

GPU Utilization

Cost Optimization

---

# Practical Labs

1. Fine-tune a sentiment analysis model using BERT.
2. Fine-tune a small language model (e.g., Llama/Qwen family) with LoRA on a custom instruction dataset.
3. Perform QLoRA fine-tuning on a single GPU.
4. Build an instruction-following chatbot using SFT.
5. Compare Full Fine-Tuning, LoRA, and QLoRA on the same dataset.
6. Fine-tune a model for domain-specific Q&A (e.g., legal or healthcare).
7. Preference-tune a model using DPO with chosen/rejected response pairs.
8. Evaluate the fine-tuned model using both automated metrics and LLM-as-a-Judge.
9. Merge adapters and deploy the model with vLLM or Ollama.
10. Track experiments with MLflow or Weights & Biases and compare hyperparameters.

# Prerequisites

Before starting fine-tuning, you should already be comfortable with:

* Python programming
* Linear algebra (vectors, matrices, matrix multiplication)
* Calculus basics (gradients and derivatives)
* Probability and statistics fundamentals
* Deep learning concepts (loss functions, optimizers, backpropagation)
* PyTorch
* NLP fundamentals (tokenization, embeddings, attention)
* Transformer architecture
* Hugging Face ecosystem (Transformers, Datasets, Tokenizers)

---

This syllabus follows a logical progression from neural network fundamentals through transformer concepts, parameter-efficient fine-tuning (LoRA/QLoRA), supervised and preference tuning, evaluation, and production deployment, making it suitable for both interview preparation and real-world LLM engineering.
