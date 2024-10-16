import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

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

    # Replace the old extract_average_color method with:
    def analyze_room(self, image_path: str) -> Dict[str, List[str]]:
        """Analyze room image for average color and dominant colors, along with detected objects."""
        image = cv2.imread(image_path)
        avg_color = self.extract_average_color(image)
        dominant_colors = self.extract_dominant_colors(image)
        detected_objects = self.detect_objects(image)
        
        return {'average_color': avg_color, 'dominant_colors': dominant_colors, 'detected_objects': detected_objects}

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
    def extract_dominant_colors(self, image, k=3) -> List[str]:
        """Extract dominant colors using k-means clustering."""
        pixels = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        dominant_colors_hex = [
            '#{:02x}{:02x}{:02x}'.format(int(color[2]), int(color[1]), int(color[0]))
            for color in dominant_colors
        ]
        return dominant_colors_hex
    
    def calculate_multi_object_match(self, detected_objects: List[str], furniture_title: str) -> float:
        # Initialize match score
        match_score = 0.0
        
        # Check for complementary furniture based on detected objects
        for obj in detected_objects:
            if obj.lower() in furniture_title.lower():
                match_score += 0.5  # Adjust weights as necessary
            elif self.is_complementary(obj, furniture_title):
                match_score += 0.3  # Complementary furniture like coffee table + sofa

        # Normalize the score based on the number of detected objects
        return match_score / max(1, len(detected_objects))
    
    def extract_furniture_size(self, furniture_details: str) -> Optional[Tuple[float, float, float]]:
        """Extract the dimensions (width, height, depth) from the furniture details using regex."""
        size_pattern = r'(\d+\.?\d*)\s*(cm|in|ft)\s*x\s*(\d+\.?\d*)\s*(cm|in|ft)\s*x\s*(\d+\.?\d*)\s*(cm|in|ft)'
        match = re.search(size_pattern, furniture_details)
        if match:
            width = float(match.group(1))
            height = float(match.group(3))
            depth = float(match.group(5))
            return width, height, depth
        return None

    def calculate_size_match(self, room_size: Optional[float], furniture_details: str) -> float:
        if room_size:
            furniture_size = self.extract_furniture_size(furniture_details)
            if furniture_size:
                # Assume room_size is in square feet and compare
                furniture_area = furniture_size[0] * furniture_size[2]  # Width x Depth
                if furniture_area <= room_size:
                    return 1.0  # Perfect fit
                elif furniture_area <= room_size * 1.2:  # Slightly larger
                    return 0.7  # Okay fit
                else:
                    return 0.3  # Doesn't fit well
        return 0.0  # No match
    
    def calculate_color_similarity(self, room_color: str, furniture_details: str) -> float:
        """Calculate color similarity between the room color (hex) and the furniture description."""
        # Check if the color is mentioned in the furniture details (simplistic)
        if room_color.lower() in furniture_details.lower():
            return 1.0  # Perfect match
        return 0.0  # No match
    
    def recommend_furniture(self, room_analysis: Dict[str, List[str]], furniture_categories: List[str], 
                        avg_color_weight=0.2, dominant_color_weight=0.2, description_weight=0.4, 
                        object_weight=0.1, size_weight=0.1) -> List[Dict[str, str]]:
        recommended_items = []
        self.dataset['combined'] = self.dataset['title'].fillna('') + ' ' + self.dataset['categories'].fillna('')

        for category in furniture_categories:
            category_items = self.dataset[self.dataset['combined'].str.contains(category, case=False)]
            if not category_items.empty:
                for _, item in category_items.iterrows():
                    # Calculate average color similarity
                    avg_color_similarity = self.calculate_color_similarity(room_analysis['average_color'], item['details'])

                    # Calculate dominant colors similarity (assumes the dominant colors are a list)
                    dominant_color_similarities = [
                        self.calculate_color_similarity(color, item['details']) for color in room_analysis['dominant_colors']
                    ]
                    dominant_color_similarity = max(dominant_color_similarities)  # Use the best match from dominant colors

                    # Calculate description, object, and size similarities
                    description_similarity = self.calculate_advanced_similarity(
                        room_analysis['detected_objects'], item['details'], item['description']
                    )
                    detected_object_similarity = self.calculate_multi_object_match(room_analysis['detected_objects'], item['title'])
                    size_fit = self.calculate_size_match(room_analysis.get('room_size'), item['details'])

                    # multi_object_match_score = self.calculate_multi_object_match(room_analysis['detected_objects'], item['title'])

                    # Combine the similarities with the specified weights
                    overall_similarity = (
                        avg_color_weight * avg_color_similarity +
                        dominant_color_weight * dominant_color_similarity +
                        description_weight * description_similarity +
                        object_weight * detected_object_similarity +
                        size_weight * size_fit
                    )

                    recommended_items.append({
                        'Title': item['title'],
                        'Details': item['details'],
                        'Similarity': overall_similarity
                    })

        # Sort items by similarity and return the top recommendations
        recommended_items.sort(key=lambda x: x['Similarity'], reverse=True)
        return recommended_items[:10]


# Example usage
if __name__ == "__main__":
    dataset_path = "data_for_test_2_1.csv"  # Path to your dataset
    system = FurnitureRecommendationSystem(dataset_path)

    image_path = "img3.jpg"  # Path to the room image
    room_analysis = system.analyze_room(image_path)
    print(room_analysis)

    furniture_categories = ['bed','lamp']  # Example categories
    recommendations = system.recommend_furniture(room_analysis, furniture_categories)

    # Output recommendations
    for rec in recommendations:
        print(f"Title: {rec['Title']}, Details: {rec['Details']}, Similarity: {rec['Similarity']:.4f}")
