## LLM Balanced Training & Inference Pipeline

This repository contains the code and configuration for training the Large Language Models Qwen3 8B, Deepseek LLM 7B and Llama3 8B Instruct using 4 financial sentiment datasets. It covers data preprocessing, model configuration, training, inference and evaluation.

## 📁 Project Structure

```
.
├── data/                   # Raw and preprocessed datasets
├── training/               # Training configuration files and pipeline
├── inference/              # Inference pipeline
└── README.md               # Project overview and instructions
```

## 🚀 Features

### Training

- Custom balanced training pipeline adapted for the 5 financial sentiment datasets. It counterweights outsampled 
- Prompt template for each model
- Training config for a single A100 40GB GPU
- LORA/PEFT training w/ 4-bit weight quantization and bf16 computing
- Training is performed with 6 different data % splits [5, 10, 20, 40, 75, 100]

### Inference

- Inference of either the base model in 0 or more shot learning or the finetuned models in ### Training
- This pipeline supports the 3 models, comment at the end of inference/main.py to run inference in all or specific models and datasets, as well as shots learning or evaluate the sft models. You can select the number of shots, however a high number will cause longer prompts which can cause memory overload or make it difficult to parse the output label
- Follows the same prompt format and quantization for each model as the training phase
- Performs inference and evaluation saving predicted outputs and metrics

## ⚙️ Setup

```bash
git clone https://github.com/NLPforFinance/llm-training-inference.git
cd llm-train-inference-main
```

## 🏋️ Training

```bash
export CUDA_VISIBLE_DEVICES=0 && python training/deepseek-train/main.py
export CUDA_VISIBLE_DEVICES=0 && python training/llama-train/main.py
export CUDA_VISIBLE_DEVICES=0 && python training/qwen-train/main.py
```



## 📊 Inference & Evaluation

```bash
export CUDA_VISIBLE_DEVICES=0 && python inference/main.py
```
