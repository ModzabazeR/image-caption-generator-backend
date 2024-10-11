import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from PIL import Image
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

nltk.download('punkt')

# 1. Set up the environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load captions
captions_file = "../AI Project/flickr8k/captions.txt"
image_dir = "../AI Project/flickr8k/Images"

# Configuration for PEFT
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "self.query",
        "self.key",
        "self.value",
        "output.dense",
        "self_attn.qkv",
        "self_attn.projection",
        "mlp.fc1",
        "mlp.fc2",
    ],
)

# Load model and processor
model_id = "Salesforce/blip-image-captioning-base"
model = AutoModelForVision2Seq.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Apply PEFT to the model
model = get_peft_model(model, config)
model.print_trainable_parameters()

captions_dict = {}
with open(captions_file, 'r') as f:
  lines = f.readlines()
  for line in lines[1:]:
        parts = line.strip().split(',', 1)
        if len(parts) == 2:
            image_file, caption = parts
            if image_file not in captions_dict:
                captions_dict[image_file] = caption
            # captions_dict[image_file].append(caption)
            
class ImageCaptioningDataset(Dataset):
    def __init__(self, image_dir, captions_dict, processor):
        self.image_dir = image_dir
        self.captions_dict = captions_dict
        self.image_files = list(captions_dict.keys())
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        caption = self.captions_dict[image_file]
        # caption = captions[0]  # Using the first caption for simplicity

        encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = caption
        return encoding

def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

# Create dataset
full_dataset = ImageCaptioningDataset(image_dir, captions_dict, processor)
print(full_dataset[0])

# Split the dataset into training and validation sets
train_size = int(0.85 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=10, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=10, collate_fn=collate_fn)

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Training loop
model.train()
for epoch in range(20):  # Adjust the number of epochs as needed
    print(f"Epoch {epoch + 1}/{40}")
    epoch_loss = 0
    for batch in train_dataloader:
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )

        loss = outputs.loss
        # loss = loss  # Convert loss to torch.float32

        # Check for NaN in loss
        if torch.isnan(loss):
            print("NaN detected in loss. Skipping batch.")
            continue

        epoch_loss += loss.item()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained('caption')