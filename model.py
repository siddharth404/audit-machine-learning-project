# Cell 1: Import Libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import os

# Cell 2: Define Utility Functions
class ComplaintsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Cell 3: Load Tokenizer and Model
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = "./finbert_trained_model.pkl"

# Load or Train Model
if os.path.exists(model_path):
    print("Loading pickled model...")
    model = load_model(model_path)
else:
    print("Training a new model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Set num_labels=2 for binary classification
        ignore_mismatched_sizes=True  # Ignore size mismatch for classifier layer
    )

    # Cell 4: Load and Prepare Data
    # Load data (assuming CFPB dataset downloaded as 'complaints.csv')
    df = pd.read_csv("../data/complaints.csv")
    df.columns = [i.lower().replace(" ", "_") for i in df.columns]

    # Filter for insurance-related complaints
    insurance_data = df[df["product"].str.contains("Insurance", na=False)]

    # Define binary labels
    # Here, the binary classification is whether the complaint is insurance-related (1) or not (0)
    df['is_insurance_related'] = df['product'].apply(lambda x: 1 if "Insurance" in str(x) else 0)

    # Extract relevant data for training
    texts = df["consumer_complaint_narrative"].fillna("")
    labels = df['is_insurance_related'].tolist()

    # Tokenize the data
    inputs = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=512,  # Adding max_length to truncate sequences properly
        return_tensors="pt"
    )

    # Cell 5: Prepare Dataset and Train Model
    dataset = ComplaintsDataset(inputs, labels)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    # Save the model to a pickle file after training
    save_model(model, model_path)

print("Model is ready for inference.")
