# LLM Instruction-Tuning for Therapist Chatbot

This project demonstrates **LLM Instruction-Tuning** using custom text data for a chatbot focused on providing psychological counseling. During the training process, `train/loss` and `eval/loss` were recorded and visualized using **WandB**.

## Chatbot Concept

- **Name**: Comfort Zone (Therapist Chatbot)
- **Industry**: Psychological Counseling
- **Target Audience**: Individuals seeking advice or emotional support, irrespective of age or gender.
- **Problem**: Many people cannot access counseling due to time, location, or cost constraints.
- **Objective**: To simulate real conversations with a counselor, providing emotional support and psychological feedback to users.

## Instruction-Tuning Steps

### 1. **Data Preparation**
- **Corpus Data**: Used `corpus.json`, which contains dialogues between `user` and `therapist`.
- **Data Split**: Divided into **80% training** and **20% validation** datasets.

### 2. **Model and Tokenizer**
- **Model**: Used `facebook/opt-350m`, a Causal Language Model.
- **Tokenizer**: Leveraged the corresponding tokenizer from `facebook/opt-350m`.

### 3. **Fine-Tuning Process**
- **Trainer**: Used Hugging Face's `SFTTrainer` for training and evaluation. Loss values (`train/loss` and `eval/loss`) were logged to WandB at each step and epoch.
- **Real-time Logging**: Monitored loss metrics in real-time via WandB.

### 4. **Results**
- **Train Loss**: Showed consistent decrease, indicating the model's learning progression.

- **Eval Loss**: Decreased during most epochs but slightly increased in later epochs, suggesting potential **overfitting**.

## Tools and Environment
- **Model**: Facebook OPT-350M (`AutoModelForCausalLM`)
- **Framework**: Hugging Face Transformers, Datasets, WandB
- **Data**: User-therapist dialogue dataset (`corpus.json`)
- **Environment**: GPU-based training, **10 Epochs**
- **Batch Size**: 8

### WandB Logs
- **Train Loss Log**: [Train Loss Link](https://api.wandb.ai/links/example-train-loss)
- **Eval Loss Log**: [Eval Loss Link](https://api.wandb.ai/links/example-eval-loss)
