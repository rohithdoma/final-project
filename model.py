!pip install -U transformers datasets accelerate -q

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd

from google.colab import files
uploaded = files.upload()

mport pandas as pd

# Load the CSV with error handling
df = pd.read_csv("Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11 (1).csv",
                 on_bad_lines='skip',  # Skip corrupted rows
                 engine='python')      # Use the more flexible parser

# Keep only the useful columns
df = df[['instruction', 'response']].dropna()

# Show the first few rows
df.head()

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

 Re-run this if tokenized_dataset is missing
 
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./flan-t5-customer-chatbot",
    per_device_train_batch_size=8,
    num_train_epochs=1,  # Reduced for speed
    learning_rate=5e-5,
    logging_steps=20,
    save_steps=100,
    save_total_limit=2,
    weight_decay=0.01,
    report_to="none"




from datasets import Dataset


df = df.rename(columns={'instruction': 'input_text', 'response': 'target_text'})
dataset = Dataset.from_pandas(df[['input_text', 'target_text']])

# Preprocessing function
def preprocess(example):
    inputs = tokenizer(example['input_text'], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(example['target_text'], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess, batched=True)


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("./flan-t5-customer-chatbot")
tokenizer.save_pretrained("./flan-t5-customer-chatbot")

