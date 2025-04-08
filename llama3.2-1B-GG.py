import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
import numpy as np
import evaluate
from datasets import load_dataset


model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)


# Loading dataset from hugging face (Great Gatsby txt)
ds = load_dataset("TeacherPuffy/book")

# This line prints out the "train" split where each index is a line number
print(ds["train"][100])


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure pad_token_id is set

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    outputs["labels"] = outputs["input_ids"].copy()  # Set labels to be identical to input_ids
    return outputs

tokenized_datasets = ds.map(tokenize_function)
print(tokenized_datasets)


tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.with_format("torch")
print(tokenized_datasets["train"])


model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
model.config.pad_token_id = tokenizer.eos_token_id  # Update model configuration
model.resize_token_embeddings(len(tokenizer))

# Contains all hyperparameters
training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=2)

# Computes and reports metrics during training
metric = evaluate.load("accuracy")

# Calculates accuracy of the predictions
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define a callback to print training loss at the end of each epoch
class LogEpochLossCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Filter log history for entries with a loss value
        loss_logs = [log for log in state.log_history if "loss" in log]
        if loss_logs:
            last_log = loss_logs[-1]
            print(f"Epoch {state.epoch:.2f} ended with loss: {last_log['loss']:.4f}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    compute_metrics=compute_metrics,
)

trainer.add_callback(LogEpochLossCallback)

# Launch training
trainer.train()