"""
This script is for the finetuning of the downstream task. 
In our case, downstream task if a regression that given a trajectory predicts
the ETA.
In HuggingFace transformers, this can be done using: BertForSequenceClassification.from_pretrained
while specifying that there's only one label.

This code is not fully working. 

"""
from transformers import (
        BertForSequenceClassification, 
        Trainer,
        TrainingArguments, 
        BertTokenizer
        )

import torch
from torch.utils.data import (
        Dataset, 
        DataLoader, 
        random_split
        )
from sklearn.model_selection import train_test_split


class TrajectoriesDataset(Dataset):
    """Trajectories dataset."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def read_traj_data(fname, hour=None):
    trajs = []
    labels = []
    with open(fname) as f:
        for line in f:
            _ = line.strip().split(',')
            if hour is None or (hour is not None and hour == int(_[1])):
                trajs.append(_[-1])
                labels.append(float(_[2]))
            if len(labels) == 200:
                break
    return trajs, labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print('Labels:', labels)
    print('Preds:', preds)
    mape = 100 * sum([abs(labels[i]-preds[i])/labels[i] for i in
        range(len(labels))])/len(labels)
    return {
        'mape': mape
    }

if __name__ == "__main__":
    # Prepare dataset
    trajs, labels = read_traj_data(fname='./data/processed/trips_h3_10.txt', 
            hour=8)

    train_texts, test_texts, train_labels, test_labels = train_test_split(trajs, labels, test_size=.2)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    tokenizer = BertTokenizer.from_pretrained('./models/215000')
    #train_encodings = tokenizer(train_texts, truncation=True, padding=True,
    #        max_length=512, pad_to_max_length=True)
    print("Train text:", train_texts[0])
    train_encodings = tokenizer(train_texts, padding=True,
            max_length=512, pad_to_max_length=True)
    val_encodings = tokenizer(val_texts, padding=True, max_length=512, 
            pad_to_max_length=True)
    test_encodings = tokenizer(test_texts, padding=True, 
            max_length=512, pad_to_max_length=True)

    train_dataset = TrajectoriesDataset(train_encodings, train_labels)
    val_dataset = TrajectoriesDataset(val_encodings, val_labels)
    test_dataset = TrajectoriesDataset(test_encodings, test_labels)
    print('Train Sample')
    for i, x in enumerate(train_dataset):
        print(len(train_dataset), i, x)
        break


    # Load model
    # Sofiane: I"m fixing the number of labels to 1 to tell transformers that
    # we're doing regression not classification. You can find this in the
    # documentation of BertForSequenceClassification. However, there are two
    # ways to do it, as in line 100 and line 102. Not sure which one is better.
    # For safety, I'm doing it in both. 
    model = BertForSequenceClassification.from_pretrained("./models/215000",
            num_labels=1)
    model.config.__dict__['num_labels'] = 1
    print(model.config)
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset            # evaluation dataset
    )
    trainer.train()
    x = trainer.evaluate()
    print(x)
    trainer.save_model()
