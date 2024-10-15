# fine_tune_clip_color_compatibility.py

import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.cluster import KMeans
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
from torch.nn import CosineEmbeddingLoss

# Helper functions
def extract_dominant_color(image, k=1):
    img = np.array(image)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)
    dominant_color = kmeans.cluster_centers_.astype(int)
    return dominant_color[0]

def calculate_color_similarity(color1, color2):
    color1 = np.array(color1)
    color2 = np.array(color2)
    return np.linalg.norm(color1 - color2)

def extract_average_color(image):
    img = np.array(image)
    avg_color = img.mean(axis=(0, 1))
    return avg_color


# 1. Preprocess COCO-Stuff Dataset
def preprocess_coco_dataset():
    print("Loading COCO-Stuff dataset...")
    coco = load_dataset("cocodataset/coco-stuff")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("Extracting dominant colors from COCO-Stuff dataset...")
    coco_dominant_colors = []
    for sample in coco['train']:
        image_path = sample['file_name']
        image = Image.open(image_path)
        preprocessed_image = preprocess(image)
        dominant_color = extract_dominant_color(image)
        coco_dominant_colors.append(dominant_color)

    return coco_dominant_colors


# 2. Preprocess Color Harmony Dataset
def preprocess_color_harmony_dataset(color_harmony_data):
    # Preprocesses based on color similarity calculation
    return calculate_color_similarity


# 3. Preprocess MIT-Adobe FiveK Dataset
def preprocess_mit_adobe_dataset(mit_adobe_images):
    print("Extracting average colors from MIT-Adobe FiveK dataset...")
    mit_adobe_avg_colors = []
    for img_path in mit_adobe_images:
        image = Image.open(img_path)
        avg_color = extract_average_color(image)
        mit_adobe_avg_colors.append(avg_color)

    return mit_adobe_avg_colors


# 4. Fine-Tuning CLIP for Color Compatibility
class FurnitureColorDataset(Dataset):
    def __init__(self, furniture_items, room_colors):
        self.furniture_items = furniture_items
        self.room_colors = room_colors

    def __len__(self):
        return len(self.furniture_items)

    def __getitem__(self, idx):
        item = self.furniture_items[idx]
        room_color = self.room_colors[idx]
        return item, room_color

def fine_tune_clip_model(model, processor, furniture_items, room_colors):
    dataset = FurnitureColorDataset(furniture_items, room_colors)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    loss_fn = CosineEmbeddingLoss()

    print("Starting fine-tuning of CLIP for color compatibility...")
    for epoch in range(5):
        for furniture_item, room_color in dataloader:
            # Encode text and image
            inputs = processor(text=[furniture_item], images=room_color, return_tensors="pt", padding=True)
            outputs = model(**inputs)

            # Calculate loss (Cosine similarity loss for color compatibility)
            cosine_sim_loss = loss_fn(outputs.text_embeds, outputs.image_embeds, torch.ones_like(outputs.text_embeds))

            # Backpropagation
            optimizer.zero_grad()
            cosine_sim_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {cosine_sim_loss.item()}")


# 5. Save the Fine-Tuned Model
def save_fine_tuned_model(model, processor, save_directory):
    print(f"Saving fine-tuned model to {save_directory}...")
    model.save_pretrained(save_directory)
    processor.save_pretrained(save_directory)
    print(f"Model successfully saved at {save_directory}")


# 6. Inference: Recommend Furniture Based on Room Background Colors
def recommend_furniture_based_on_color(room_color, furniture_items, model, processor):
    recommendations = []
    for item in furniture_items:
        image = Image.open(item)
        inputs = processor(text=["furniture"], images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        similarity = np.linalg.norm(room_color - extract_dominant_color(image))
        recommendations.append((item, similarity))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:5]  # Return top 5 recommendations


if __name__ == "__main__":
    # Load pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load datasets and preprocess
    coco_dominant_colors = preprocess_coco_dataset()

    # Preprocess the Color Harmony Dataset (sample input)
    color_harmony_data = [
        {"dominant_color": [255, 0, 0], "harmonious_color": [0, 0, 255]},
        {"dominant_color": [0, 255, 0], "harmonious_color": [255, 255, 0]}
    ]
    preprocess_color_harmony_dataset(color_harmony_data)

    # Preprocess MIT-Adobe FiveK Dataset (sample input)
    mit_adobe_images = ["path_to_image1.jpg", "path_to_image2.jpg"]
    preprocess_mit_adobe_dataset(mit_adobe_images)

    # Fine-tune CLIP model
    furniture_items = ["path_to_furniture_image1.jpg", "path_to_furniture_image2.jpg"]
    room_colors = coco_dominant_colors[:len(furniture_items)]
    fine_tune_clip_model(model, processor, furniture_items, room_colors)

    # Save fine-tuned model
    save_directory = "./fine_tuned_clip_model"
    save_fine_tuned_model(model, processor, save_directory)

    # Inference: Recommend furniture based on color compatibility
    room_color = [255, 0, 0]  # Example dominant color
    top_furniture = recommend_furniture_based_on_color(room_color, furniture_items, model, processor)
    print("Top Furniture Recommendations:", top_furniture)
