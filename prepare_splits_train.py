from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification, EarlyStoppingCallback, IntervalStrategy
import numpy as np
import pandas as pd
import pickle
from datasets import DatasetDict, Dataset, ClassLabel, load_dataset, load_metric, DownloadMode
import wandb

wandb.login(key="your_key")

with open("new_pos_train_data.pickle", 'rb') as file:
    df = pickle.load(file)

with open("new_pos_encoding.pickle", 'rb') as file:
    encoding_dict = pickle.load(file)

num_labels = len(encoding_dict)

text_column_name = 'words'
label_column_name = 'pos'

config = AutoConfig.from_pretrained('ai4bharat/indic-bert', num_labels=num_labels, finetuning_task='ner')
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
model = AutoModelForTokenClassification.from_pretrained('ai4bharat/indic-bert', num_labels=num_labels )

# Tokenize all texts and align the labels with them.
padding = "max_length"
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        max_length=512,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        # print('=====')
        # print('{} {}'.format(i,label)) #ak
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Assuming you have a DataFrame called 'df' with columns 'tokens', 'pos_tags', and 'split'

# Create a dictionary to store the dataset splits
dataset_dict = DatasetDict()

# Select the columns of interest
selected_columns = ["words", "pos"]

# Create a dataset for the training split
train_dataset = Dataset.from_pandas(df[df['split'] == 'train'][selected_columns])
train_dataset = train_dataset.remove_columns("__index_level_0__")  # Remove the '__index_level_0__' column
dataset_dict['train'] = train_dataset

# Create a dataset for the test split
test_dataset = Dataset.from_pandas(df[df['split'] == 'test'][selected_columns])
test_dataset = test_dataset.remove_columns("__index_level_0__")  # Remove the '__index_level_0__' column
dataset_dict['test'] = test_dataset

# Create a dataset for the validation split
val_dataset = Dataset.from_pandas(df[df['split'] == 'validation'][selected_columns])
val_dataset = val_dataset.remove_columns("__index_level_0__")  # Remove the '__index_level_0__' column
dataset_dict['validation'] = val_dataset

# Print the dataset structure
#print(dataset_dict)

train_dataset = dataset_dict["train"]
train_dataset = train_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    num_proc=4,
    load_from_cache_file=True,
    desc="Running tokenizer on train dataset",
)

eval_dataset = dataset_dict["validation"]
eval_dataset = eval_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    num_proc=4,
    load_from_cache_file=True,
    desc="Running tokenizer on Validation dataset",
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# Metrics
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value
    return final_results

# args=TrainingArguments(output_dir='output_dir',max_steps=5)
args=TrainingArguments(
    output_dir='output_dir',
    fp16=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,)

# Initialize our Trainer
# early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)
# args.metric_for_best_model = "f1"
# args.load_best_model_at_end = True
# args.evaluation_strategy = IntervalStrategy.STEPS
# args.eval_steps = args.save_steps
# args.greater_is_better = True

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # callbacks=[early_stopping_callback],
    args=args,
)

trainer.args

train_result = trainer.train()
metrics = train_result.metrics
