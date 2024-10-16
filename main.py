import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import re
from sentence_transformers import SentenceTransformer


# Constants
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Constants for color matching
PREDEFINED_COLORS = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'grey': (128, 128, 128),
    'brown': (165, 42, 42),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'pink': (255, 192, 203),
    'beige': (245, 222, 179),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255)
}

# Furniture dataset
class FurnitureRecommendationSystem:
    def __init__(self, dataset_path: str):
        self.dataset = pd.read_csv(dataset_path)
        self.scaler = StandardScaler()
        # Load the SentenceTransformer model for generating text embeddings
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def analyze_room(self, image_path: str) -> Dict[str, List[str]]:
        # Load and analyze the room image
        image = cv2.imread(image_path)
        avg_color = self.extract_average_color(image)
        detected_objects = self.detect_objects(image)
        return {'average_color': avg_color, 'detected_objects': detected_objects}

    def extract_average_color(self, image) -> str:
        # Calculate the average color in RGB space
        avg_color = cv2.mean(image)[:3]  # Get the mean color in BGR
        avg_color_hex = '#{:02x}{:02x}{:02x}'.format(int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))
        return avg_color_hex

    def detect_objects(self, image) -> List[str]:
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = net.getLayerNames()
        
        # Adjusting this line to accommodate the possible return type of getUnconnectedOutLayers()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

        # Prepare image for detection
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        detected_classes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    detected_classes.append(COCO_CLASSES[class_id])

        return list(set(detected_classes))  # Return unique detected objects


    def recommend_furniture(self, room_analysis: Dict[str, List[str]], furniture_categories: List[str]) -> List[Dict[str, str]]:
        recommended_items = []
        # Assuming self.dataset is your DataFrame containing 'title', 'categories', and 'description'
        self.dataset['combined'] = self.dataset['title'].fillna('') + ' ' + self.dataset['categories'].fillna('')

        for category in furniture_categories:
            category_items = self.dataset[self.dataset['combined'].str.contains(category, case=False)]
            if not category_items.empty:
                for _, item in category_items.iterrows():
                    # Calculate similarity based on multiple criteria
                    color_similarity = self.calculate_similarity(room_analysis['average_color'], item['details'])
                    # description_similarity = self.calculate_similarity(room_analysis['average_color'], item['description'])
                    description_similarity = self.calculate_advanced_similarity(
                        room_analysis['detected_objects'], item['details'], item['description']
                    )
                    # Additional features
                    detected_object_similarity = self.calculate_object_match(room_analysis['detected_objects'], item['title'])
                    size_fit = self.calculate_size_match(room_analysis.get('room_size'), item['details'])  # Optional

                    # Aggregate the scores (You can use different weights for each criterion)
                    overall_similarity = (0.4 * color_similarity + 0.4 * description_similarity +
                                        0.1 * detected_object_similarity + 0.1 * size_fit)

                    recommended_items.append({
                        'Title': item['title'],
                        'Details': item['details'],
                        'Similarity': overall_similarity
                    })

        # Sort by overall similarity score
        recommended_items.sort(key=lambda x: x['Similarity'], reverse=True)
        return recommended_items[:10]  # Return top 10 recommendations

    def calculate_similarity(self, room_color: str, furniture_details: str) -> float:
        # Example similarity calculation based on color and text
        room_color_rgb = self.hex_to_rgb(room_color)
        furniture_color = self.extract_color_from_description(furniture_details)
        color_distance = self.calculate_color_distance(room_color_rgb, furniture_color)
        # print(room_color,furniture_color,color_distance)
        return 1 / (1 + color_distance)  # Inverse distance as a similarity score

    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def extract_color_from_description(self, description: str) -> Tuple[int, int, int]:
        # Find colors in the description using predefined colors
        for color_name, rgb in PREDEFINED_COLORS.items():
            if re.search(r'\b' + re.escape(color_name) + r'\b', description, re.IGNORECASE):
                return rgb  # Return RGB value if color found
        return (255, 255, 255)  # Default to white if no color found

    def calculate_color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        return np.linalg.norm(np.array(color1) - np.array(color2))
    
    # New method to calculate object match
    def calculate_object_match(self, detected_objects: List[str], furniture_title: str) -> float:
        # Check if the furniture is related to detected objects (if sofa is detected, recommend chairs)
        match_score = 0.0
        for obj in detected_objects:
            if obj.lower() in furniture_title.lower():
                match_score = 1.0  # Exact match
            elif self.is_complementary(obj, furniture_title):
                match_score = 0.7  # Complementary furniture (sofa + coffee table)
        return match_score

    def is_complementary(self, object_name: str, furniture_title: str) -> bool:
        # Basic logic to define complementary furniture (can be expanded with domain knowledge)
        complementary_map = {
            'sofa': ['coffee table', 'side table'],
            'bed': ['nightstand', 'dresser'],
            'desk': ['chair', 'bookshelf']
        }
        return any(item in furniture_title.lower() for item in complementary_map.get(object_name.lower(), []))

    def calculate_size_match(self, room_size: Optional[float], furniture_details: str) -> float:
        # This is a placeholder for matching the room size with furniture dimensions (requires actual data)
        if room_size and 'dimensions' in furniture_details:
            # Parse dimensions from details and calculate whether the furniture fits
            furniture_size = self.extract_furniture_size(furniture_details)
            if furniture_size <= room_size:
                return 1.0  # Perfect fit
            else:
                return 0.5  # Not ideal, but still might fit
        return 0.0  # No size match
    
    def calculate_advanced_similarity(self, detected_objects: List[str], details: str, description: str) -> float:
        # Combine details and description into one text block for semantic similarity
        furniture_text = details + ' ' + description
        room_context = ' '.join(detected_objects)  # Join detected objects as room context text

        # Generate embeddings for both texts (room context and furniture text)
        room_embedding = self.model.encode([room_context])
        furniture_embedding = self.model.encode([furniture_text])

        # Calculate cosine similarity between the embeddings
        similarity_score = cosine_similarity(room_embedding, furniture_embedding)[0][0]
        return similarity_score

# Example usage
if __name__ == "__main__":
    dataset_path = "data_for_test_2_1.csv"  # Path to your dataset
    system = FurnitureRecommendationSystem(dataset_path)

    image_path = "img1.jpg"  # Path to the room image
    room_analysis = system.analyze_room(image_path)
    print(room_analysis)

    furniture_categories = ['bed']  # Example categories
    recommendations = system.recommend_furniture(room_analysis, furniture_categories)

    # Output recommendations
    for rec in recommendations:
        print(f"Title: {rec['Title']}, Details: {rec['Details']}, Similarity: {rec['Similarity']:.4f}")
