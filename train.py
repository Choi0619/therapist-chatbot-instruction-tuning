import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from trl import SFTConfig, SFTTrainer
import wandb
from transformers import TrainerCallback

# Initialize WandB
wandb.init(project="therapist-chatbot", name="fine-tuning")

# Load corpus.json dataset
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Prepare input-output pairs
data_pairs = []
for i in range(0, len(corpus) - 1, 2):  # Iterate through user-therapist pairs
    if corpus[i]['role'] == 'user' and corpus[i + 1]['role'] == 'therapist':
        input_text = corpus[i]['content']  # User input
        output_text = corpus[i + 1]['content']  # Therapist response
        data_pairs.append({"input": input_text, "output": output_text})

# Split into training and validation sets (80-20 ratio)
train_data, val_data = train_test_split(data_pairs, test_size=0.2, random_state=42)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# Define preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(examples['input'], max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples['output'], max_length=256, truncation=True, padding="max_length").input_ids

    # Set <pad> tokens to -100 to ignore them during loss computation
    labels = [[(label if label != tokenizer.pad_token_id else -100) for label in label_list] for label_list in labels]

    inputs["labels"] = labels
    return inputs

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Use DataCollatorWithPadding for dynamic batching
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define SFT configuration and trainer
sft_config = SFTConfig(
    output_dir="./results",
    eval_strategy="epoch",  # Evaluate after each epoch
    logging_strategy="steps",  # Log after a set number of steps
    logging_steps=10,  # Log every 10 steps
    eval_steps=10,  # Evaluate every 10 steps
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,
    num_train_epochs=10,  # Set to 10 epochs
    save_total_limit=1,
    fp16=False,  # Disable FP16
    run_name="therapist-fine-tuning-run"  # WandB run name
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=sft_config,
    data_collator=collator,
)

# Define TrainerCallback for WandB integration
class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs)

    def on_epoch_end(self, args, state, control, **kwargs):
        # Log loss to WandB at the end of each epoch
        if len(state.log_history) > 0:
            for log in state.log_history:
                if "loss" in log:
                    logs = {"train/loss": log["loss"], "train/epoch": state.epoch}
                    wandb.log(logs)

# Add WandB callback to the trainer
trainer.add_callback(WandbCallback)

# Start training
train_result = trainer.train()

# Evaluate on validation dataset
eval_metrics = trainer.evaluate()

# Log evaluation results to WandB
wandb.log({"eval/loss": eval_metrics.get('eval_loss', 0), "eval/epoch": eval_metrics.get('epoch', 0)})

# Save the fine-tuned model
trainer.save_model("./fine_tuned_therapist_chatbot")

# Check training log history
df = pd.DataFrame(trainer.state.log_history)
print(df)  # Print log history to verify loss tracking

# Log training and evaluation results
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)

trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

# Finish WandB logging
wandb.finish()
