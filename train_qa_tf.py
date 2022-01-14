import subprocess
import sys
import os
import argparse
import json

EVAL_RATIO = 0.2
MODEL = "raduion/bert-medium-luxembourgish"
try:
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoConfig
    from transformers import DataCollatorWithPadding
    from transformers import set_seed
    import tensorflow as tf
    from datasets import load_dataset
    from datasets import ClassLabel
    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    from tensorflow.keras.optimizers.schedules import PolynomialDecay
    from tensorflow.keras.optimizers import Adam


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
    return parser.parse_args()


def tokenize(data):
    tokens = tokenizer(data["question"], truncation=True, max_length=1024, padding=True)
    tokens['label'] = labels.str2int(data['label'])
    return tokens


if __name__ == '__main__':
    args = parse_args()
    set_seed(42)
    data = json.load(open(args.qa_set, encoding="utf-8"))

    raw_labels = list(set([x['label'] for x in data['data']]))
    labels = ClassLabel(num_classes=len(raw_labels), names=raw_labels)
    label2id = {raw_labels[k]: k for k in range(len(raw_labels))}
    id2label = {k: raw_labels[k] for k in range(len(raw_labels))}

    dataset = load_dataset('json', data_files=args.qa_set, field='data')['train']

    epochs = [10, 50, 100, 200, 400]
    b_sizes = [8, 16, 32]
    for epoch in epochs:
        for b_size in b_sizes:
            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            config = AutoConfig.from_pretrained(MODEL, label2id=label2id, id2label=id2label)
            model = TFAutoModelForSequenceClassification.from_pretrained(MODEL, config=config)
            data_collator = DataCollatorWithPadding(tokenizer, return_tensors='tf')

            X = dataset.train_test_split(test_size=EVAL_RATIO)
            data_encoded = X.map(tokenize)

            tf_train_dataset = data_encoded["train"].to_tf_dataset(
                columns=["attention_mask", "input_ids"],
                label_cols=["labels"],
                shuffle=True,
                collate_fn=data_collator,
                batch_size=b_size,
            )

            tf_validation_dataset = data_encoded["test"].to_tf_dataset(
                columns=["attention_mask", "input_ids"],
                label_cols=["labels"],
                shuffle=False,
                collate_fn=data_collator,
                batch_size=b_size,
            )
            NAME = m + "-epochs-" + str(epoch) + "-b_size-" + str(b_size)

            # The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
            # by the total number of epochs
            num_train_steps = len(tf_train_dataset) * epoch
            lr_scheduler = PolynomialDecay(
                initial_learning_rate=1e-5, end_learning_rate=0.0, decay_steps=num_train_steps
            )

            opt = Adam(learning_rate=lr_scheduler)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
            tb_callback = tf.keras.callbacks.TensorBoard('./logs/' + NAME, update_freq=1)
            model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=epoch, callbacks=[tb_callback])
