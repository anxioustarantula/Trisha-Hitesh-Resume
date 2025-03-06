!pip install transformers datasets torch scikit-learn

from datasets import load_dataset
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from transformers import T5Tokenizer

# Initialize the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokens list
tokens = ['▁Core', 'ference', '▁resolution', ':', '▁Mark', '▁told', '▁Pete', '▁many',
          '▁lies', '▁about', '▁himself', ',', '▁which', '▁Pete', '▁included', '▁in',
          '▁his', '▁book', '.', '▁He', '▁should', '▁have', '▁been', '▁more', '▁skeptical',
          '.', '▁Does', '▁', "'", 'He', "'", '▁refer', '▁to', '▁', "'", 'Mark', "'", '?']

# Convert tokens to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Print token IDs
print(token_ids)
# Load the WSC dataset
def preprocess_wsc_data():
    # Download the WSC dataset from the SuperGLUE benchmark
    wsc = load_dataset("super_glue", "wsc")

    # Preprocess the data
    def preprocess(ex):
        sentence = ex['text']
        pronoun = ex['span2_text']
        candidate = ex['span1_text']
        label = "true" if ex['label'] == 1 else "false"

        # Formatting input text for T5
        input_text = f"resolve: {sentence}\nPronoun: {pronoun}\nCandidate: {candidate}"
        target_text = label

        return {"input_text": input_text, "target_text": target_text}

    # Apply the preprocessing function and remove irrelevant columns
    train_data = wsc['train'].map(preprocess, remove_columns=['text', 'span1_text', 'span2_text', 'label'])
    val_data = wsc['validation'].map(preprocess, remove_columns=['text', 'span1_text', 'span2_text', 'label'])

    return train_data, val_data

train_data, val_data = preprocess_wsc_data()


train_df = train_data.to_pandas()
train_df
val_df = val_data.to_pandas()
val_df

import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained('t5-small')  # or 't5-large' depending on your choice
#model = T5ForConditionalGeneration.from_pretrained('t5-small')
from transformers import T5Config

# Create a custom T5 configuration
config = T5Config(
    num_layers=6,          # Encoder and decoder layers
    num_decoder_layers=6,  # Decoder layers (if different)
    d_model=512,           # Hidden size
    dropout_rate=0.1      # Dropout
)

# Initialize model with custom config
model = T5ForConditionalGeneration(config=config)
# Move model to the correct device (e.g., CUDA if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Assuming 'train_data' and 'val_data' are already preprocessed datasets

# Create a clean and simple structure for the training data (no tokenization yet)
train_input_text = [data['input_text'] for data in train_data]
train_target_text = [data['target_text'] for data in train_data]

# Create a DataFrame to easily inspect the preprocessed data before tokenization
train_df = pd.DataFrame({
    "input_text": train_input_text,
    "target_text": train_target_text
})

# Display the first few rows to verify
train_df
from transformers import T5Tokenizer

# Load T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokenize the dataset in batches
def tokenize_data(dataset, tokenizer, max_length=512):
    # Tokenize input and target texts and pad them
    input_encodings = tokenizer(
        dataset['input_text'],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    target_encodings = tokenizer(
        dataset['target_text'],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Add the target labels to the input encodings explicitly
    input_encodings['labels'] = target_encodings['input_ids']

    # Returning the tokenized dataset
    return input_encodings

# Tokenizing the entire dataset
train_encodings = tokenize_data(train_data, tokenizer)
val_encodings = tokenize_data(val_data, tokenizer)
# # import torch

# class WSCDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __len__(self):
#         return len(self.encodings['input_ids'])

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         return item

# train_dataset = WSCDataset(train_encodings)
# val_dataset = WSCDataset(val_encodings)

class WSCDataset(Dataset):
    def __init__(self, encodings):
        """
        Initializes the dataset with tokenized encodings.
        The dataset includes input_ids, attention_mask, and labels (target_ids).
        """
        self.encodings = encodings

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        """Returns the item at the given index"""
        # For efficiency, directly return the encodings as tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

# Create the dataset instances
train_dataset = WSCDataset(train_encodings)
val_dataset = WSCDataset(val_encodings)

# Create DataLoader instances for efficient batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Example: Iterate through the data to verify
for batch in train_loader:
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch['attention_mask']
    print(input_ids.shape, labels.shape, attention_mask.shape)
    break  # Just checking the first batch
# Load the T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Check if CUDA is available and move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Verify model and device setup
print(f"Model is loaded on {device}.")

# Optionally set the model to evaluation mode if you're not training right now
model.eval()

# You can also disable gradients during evaluation for memory efficiency
# Use the model# Hyperparameters
batch_size = 8                # Batch size for training
epochs = 5                # Number of training epochs
learning_rate = 5e-4           # Learning rate for the optimizer
warmup_ratio = 0.1             # Ratio of warmup steps (10% of total steps)
max_length = 512               # Maximum length of input sequences
# gradient_accumulation_steps = 1 # Number of gradient accumulation steps (useful for simulating larger batch sizes)

# Device (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimizer and Scheduler settings
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Calculate the total number of training steps
num_training_steps = epochs * len(train_loader)
num_warmup_steps = int(warmup_ratio * num_training_steps)

# Linear scheduler with warmup
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

# # Print summary of parameters
# print(f"Training Parameters: ")
# print(f" - Batch Size: {batch_size}")
# print(f" - Epochs: {epochs}")
# print(f" - Learning Rate: {learning_rate}")
# print(f" - Warmup Steps: {num_warmup_steps}")
# print(f" - Total Training Steps: {num_training_steps}")
# print(f" - Max Sequence Length: {max_length}")in inference mode (during validation or testing):

def train_model(model, train_loader, val_loader, epochs, optimizer, scheduler, device):
    # Set the model to training mode
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        # Create a tqdm progress bar for the training loop
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for batch in train_loop:
            # Move batch data to the correct device (GPU or CPU)
            inputs = {key: val.to(device) for key, val in batch.items()}

            # Zero gradients from the previous step
            optimizer.zero_grad()

            # Forward pass through the model
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels']
            )

            # Calculate the loss and backpropagate
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Accumulate the loss for reporting
            total_loss += loss.item()

            # Update the progress bar with the current loss
            train_loop.set_postfix(loss=total_loss / (train_loop.n + 1))  # Average loss so far

        # Print average training loss for the epoch
        print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}")

        # Validation phase
        model.eval()
        total_val_loss = 0

        # No gradients needed for validation
        with torch.no_grad():
            # Create a tqdm progress bar for the validation loop
            val_loop = tqdm(val_loader, desc="Validation", unit="batch")
            for batch in val_loop:
                # Move batch data to the correct device
                inputs = {key: val.to(device) for key, val in batch.items()}

                # Forward pass (no backward pass for validation)
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['labels']
                )

                # Accumulate validation loss
                total_val_loss += outputs.loss.item()

                # Update the progress bar with the current validation loss
                val_loop.set_postfix(val_loss=total_val_loss / (val_loop.n + 1))  # Average val loss

        # Print the validation loss for the epoch
        print(f"Validation Loss: {total_val_loss / len(val_loader)}")

        # Set the model back to training mode
        model.train()

        # Optionally: Save model checkpoint after each epoch
        # torch.save(model.state_dict(), f"checkpoint_epoch_{epoch + 1}.pt")

# Run the training loop
train_model(model, train_loader, val_loader, epochs, optimizer, scheduler, device)


