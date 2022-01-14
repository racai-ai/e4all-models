import subprocess
import sys
import os
import argparse
import json

TRAINED = "model/e4a_"
EVAL_RATIO = 0.2
try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
    from transformers import TrainingArguments, Trainer, AutoConfig
    from datasets import ClassLabel
    from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset, DataCollatorWithPadding
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from transformers import set_seed


except:
    print("Required modules not installed. Installing requirements...")
    out = subprocess.check_output([sys.executable, '-m', 'pip', 'install', '-r', '../requirements.txt'])
    print(out.decode())
    print("Running the script again...")
    out = subprocess.check_output([sys.executable] + sys.argv)
    print(out.decode())
    exit()


def parse_args():
    """Parses Command line arguments
    """
    parser = argparse.ArgumentParser(description='Training script for e4a')
    parser.add_argument('-qa', '--qa_set', help='Generate data from --source folder')
    parser.add_argument('-s', '--source', help='Source folder for corpus data')
    parser.add_argument('-m', '--model', default="racai/distilbert-base-romanian-cased", help='Model to train')
    parser.add_argument('-c', '--comment', default="empty", help='Comment for training data')

    return parser.parse_args()


def retrieve_all_lines(path: str):
    r"""Retrieving all data from path and return it as array.

    Arguments:

        path_data (:obj:`str`):
          Path to the covid or authorization folder


    """
    files = [os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk(path)
             for filename in filenames]
    all_lines = []
    for file in files:
        with open(file, 'r', encoding="utf-8") as f:
            lines = (line.rstrip() for line in f)  # All lines including the blank ones
            lines = list(line for line in lines if len(line) > 25)  # Non-blank lines
            all_lines.append(lines)
    return all_lines


def tokenize(data):
    tokens = tokenizer(data["question"], truncation=True, max_length=1024, padding=True)
    tokens['label'] = labels.str2int(data['label'])
    return tokens


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


if __name__ == '__main__':
    args = parse_args()
    set_seed(42)
    data = json.load(open(args.qa_set, encoding="utf-8"))

    raw_labels = list(set([x['label'] for x in data]))
    labels = ClassLabel(num_classes=len(raw_labels), names=raw_labels)
    label2id = {raw_labels[k]: k for k in range(len(raw_labels))}
    id2label = {k: raw_labels[k] for k in range(len(raw_labels))}

    epochs = [3, 5, 10]
    b_sizes = [4, 8, 16, 32]
    for epoch in epochs:
        for b_size in b_sizes:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            config = AutoConfig.from_pretrained(args.model, label2id=label2id, id2label=id2label)
            model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            data_encoded = list(map(tokenize, data))
            X_train, X_eval = train_test_split(data_encoded, random_state=42, test_size=EVAL_RATIO)

            NAME = args.model.split("/")[1] + "-epochs-" + str(epoch) + "-b_size-" + str(b_size)
            training_args = TrainingArguments(
                output_dir='./results/' + NAME,
                learning_rate=2e-5,
                per_device_train_batch_size=b_size,
                per_device_eval_batch_size=b_size,
                num_train_epochs=epoch,
                weight_decay=0.01,
                overwrite_output_dir=True,
                evaluation_strategy='epoch',
                logging_steps=10,
                logging_strategy='steps',
                logging_dir='./logs/' + NAME
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=X_train,
                eval_dataset=X_eval,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )

            train_result = trainer.train()

            # compute train results
            metrics = train_result.metrics
            max_train_samples = len(X_train)
            metrics["train_samples"] = len(X_train)

            # save train results
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

            # compute evaluation results
            trainer.eval_dataset = X_eval
            metrics = trainer.evaluate()
            max_val_samples = len(X_eval)
            metrics["test_samples"] = len(X_eval)

            # save evaluation results
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
            # save
            trainer.save_model(TRAINED + args.source + NAME)
            tokenizer.save_pretrained(TRAINED + args.source + NAME)
            print('Finished training all...', TRAINED + args.source)
