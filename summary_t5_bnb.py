import wandb
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import BitsAndBytesConfig
import evaluate
import numpy as np
import os
from dotenv import load_dotenv, find_dotenv
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.2)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False
)

checkpoint = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_billsum = billsum.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, quantization_config=quantization_config)

os.environ["WANDB_PROJECT"] = "summary_t5_bnb"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

load_dotenv(find_dotenv())
wandb_token = os.getenv("WANDB_TOKEN")
wandb.login(key=wandb_token)

training_args = Seq2SeqTrainingArguments(
    output_dir="sum_model_t5_bnb",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    report_to="wandb"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
