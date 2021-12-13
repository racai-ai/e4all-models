import subprocess
import sys
import os

TRAINED = "model/valentin"
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


def get_lines():
    files = [os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk("autorizatii")
             for filename in filenames]
    all_lines = []
    for file in files:
        with open(file, 'r', encoding="utf-8") as f:
            lines = (line.rstrip() for line in f)  # All lines including the blank ones
            lines = list(line for line in lines if len(line) > 25)  # Non-blank lines
            all_lines.append(lines)
    return all_lines


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("racai/distilbert-base-romanian-cased")

    # text = [item for sublist in get_lines() for item in sublist]
    # with open('data.txt', 'w', encoding="utf-8") as f_output:
    #     f_output.write("\n".join(text))
    # exit(1)

    model = AutoModelForMaskedLM.from_pretrained("racai/distilbert-base-romanian-cased")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path='data.txt',
        block_size=128,
    )

    training_args = TrainingArguments(
        output_dir=TRAINED,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    print('Start a trainer...')
    # Start training
    trainer.train()

    # Save
    trainer.save_model(TRAINED)
    print('Finished training all...', TRAINED)
