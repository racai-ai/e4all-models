import subprocess
import sys
import os
import math
import argparse

TRAINED = "model/e4a_"
EVAL_RATIO = 0.1
TEST_RATIO = 0.2
try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from transformers import TrainingArguments, Trainer
    from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset
    from sklearn.model_selection import train_test_split

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
    parser.add_argument('-g', '--generate', action='store_true', help='Generate data from --source folder')
    parser.add_argument('-s', '--source', help='Source folder for corpus data')
    parser.add_argument('-m', '--model', default="racai/distilbert-base-romanian-cased", help='Model to train')
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


def create_dataset_files():
    r""" Create train tet eval split and save it in separate files

    """
    text = [item for sublist in retrieve_all_lines(args.source) for item in sublist]
    X_train, X_test = train_test_split(text, random_state=42, test_size=TEST_RATIO)
    X_train, X_eval = train_test_split(X_train, random_state=42, test_size=EVAL_RATIO)
    files = ['train', 'test', 'eval']
    dataset = [X_train, X_test, X_eval]
    for file, dataset in list(zip(files, dataset)):
        with open(file, 'w', encoding="utf-8") as f_output:
            f_output.write("\n".join(dataset))


if __name__ == '__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    if args.generate:
        create_dataset_files()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    train_data = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path='train',
        block_size=256,
    )
    eval_data = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path='eval',
        block_size=256,
    )
    test_data = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path='test',
        block_size=256,
    )

    training_args = TrainingArguments(
        output_dir='pretrain',
        overwrite_output_dir=True,
        num_train_epochs=2,
        evaluation_strategy='steps',
        eval_steps=None,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=100,
        save_steps=-1,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=test_data
    )
    print('Start initial evaluation...')
    eval_output = trainer.evaluate()
    # compute perplexity from model loss.
    perplexity = math.exp(eval_output["eval_loss"])
    print('\nInitial Perplexity: {:10,.2f}'.format(perplexity))

    # set eval_data to train
    trainer.eval_dataset = eval_data

    # Start training
    trainer.train()

    # Save
    trainer.save_model(TRAINED + args.source)
    tokenizer.save_pretrained(TRAINED + args.source)
    print('Finished training all...', TRAINED + args.source)

    # prepare to evaluate the trained model
    trainer.eval_dataset = test_data

    eval_output = trainer.evaluate()
    # compute perplexity from model loss.
    perplexity = math.exp(eval_output["eval_loss"])
    print('\nEvaluate Perplexity: {:10,.2f}'.format(perplexity))
