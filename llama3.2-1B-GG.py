import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
import numpy as np
import evaluate
from datasets import load_dataset, DatasetDict


model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)


# Loading dataset from hugging face (Great Gatsby txt)
ds = load_dataset("TeacherPuffy/book")

# Creating the train/validation/test splits
train_testvalid = ds["train"].train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)

ds_split = DatasetDict({
    "train": train_testvalid["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"]
})

print(ds_split)



tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure pad_token_id is set

def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512,
        ) 
    outputs["labels"] = outputs["input_ids"].copy()  # Set labels to be identical to input_ids
    return outputs

tokenized_datasets = ds_split.map(tokenize_function)
print(tokenized_datasets)


tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.with_format("torch")
print(tokenized_datasets["train"])


model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
model.config.pad_token_id = tokenizer.eos_token_id  # Update model configuration
model.resize_token_embeddings(len(tokenizer))

# Contains all hyperparameters
training_args = TrainingArguments(
    output_dir="test_trainer", 
    num_train_epochs=5,
    logging_strategy="epoch",
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save checkpoint at the end of each epoch
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="accuracy"  # Accuracy to determine the best model
)

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
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

trainer.add_callback(LogEpochLossCallback)

# Launch training
trainer.train()

# Evaluate on the test set
test_results = trainer.evaluate(tokenized_datasets["test"])
print(f"Test results: {test_results}")


# Save model
trainer.save_model("great_gatsby_llm")
tokenizer.save_pretrained("great_gatsby_llm")




