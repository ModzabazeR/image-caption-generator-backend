import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from PIL import Image
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from tkinter import Tk, filedialog
import random

nltk.download('punkt')

# 1. Set up the environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load captions
captions_file = "../AI Project/flickr8k/captions.txt"
image_dir = "../AI Project/flickr8k/Images"

captions_dict = {}
with open(captions_file, 'r') as f:
  lines = f.readlines()
  for line in lines:
        parts = line.strip().split(',', 1)
        if len(parts) == 2:
            image_file, caption = parts
            # if image_file not in captions_dict:
            #     captions_dict[image_file] = caption
            if image_file not in captions_dict:
                captions_dict[image_file] = [caption]
            else:
                captions_dict[image_file].append(caption)

def get_test_image_paths(img_dir):
    filenames = [
        "3259002340_707ce96858.jpg",
        "3597924257_d0da3c5fe6.jpg",
        "911795495_342bb15b97.jpg",
        "656260720_a7db4ce48b.jpg",
        "3498997518_c2b16f0a0e.jpg",
        "3711611500_ea47b58b6f.jpg",
        "895502702_5170ada2ee.jpg",
        "3550253365_27d4c303cf.jpg",
        "3425071001_e7c9809ef2.jpg",
        "3407317539_68765a3375.jpg",
    ]

    return [os.path.join(img_dir, filename) for filename in filenames]

# Load the fine-tuned model and processor
model_id = "Salesforce/blip-image-captioning-base"
model_path = 'caption'
finetuned_model = AutoModelForVision2Seq.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_id)

test_paths = get_test_image_paths(image_dir)

print(test_paths)
overall_bleu_score = 0
for file_path in test_paths:
    raw_image = Image.open(file_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")

    out = finetuned_model.generate(**inputs)
    generated_caption = processor.decode(out[0], skip_special_tokens=True)

    print(f"Reference Captions for {os.path.basename(file_path)}:")
    print(captions_dict[os.path.basename(file_path)])
    print(f"Generated Caption: {generated_caption}")

    # Tokenize the generated caption
    generated_tokens = generated_caption.lower().split()

    # Tokenize the reference captions
    reference_tokens = [caption.lower().split() for caption in captions_dict[os.path.basename(file_path)]]

    smoothing_function = SmoothingFunction().method1

    # Compute the BLEU score
    bleu_score = sentence_bleu(reference_tokens, generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
    print("BLEU-1 score:", bleu_score)

    bleu2_score = sentence_bleu(reference_tokens, generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    print("BLEU-2 score:", bleu2_score)

    bleu3_score = sentence_bleu(reference_tokens, generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
    print("BLEU-3 score:", bleu3_score)

    bleu4_score = sentence_bleu(reference_tokens, generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
    print("BLEU-4 score:", bleu4_score)

    average_bleu = (bleu_score + bleu2_score + bleu3_score + bleu4_score) / 4
    print("Average BLEU score:", average_bleu)

    overall_bleu_score += average_bleu

    print("*"*20)

overall_bleu_score /= 10
print(f"Overall BLEU score: {overall_bleu_score}")