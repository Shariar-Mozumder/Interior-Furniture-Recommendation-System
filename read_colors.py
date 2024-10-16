import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from typing import List, Tuple

def extract_exact_colors(image: np.ndarray, k: int = 5) -> Tuple[List[Tuple[str, float]]]:
    """Extract significant colors and their percentages from the image using KMeans without preprocessing."""
    
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    # Use KMeans to cluster the colors in the image
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the colors (centroids) from the clusters
    colors = kmeans.cluster_centers_.astype(int)
    
    # Count the number of pixels assigned to each cluster
    pixel_counts = Counter(kmeans.labels_)
    
    # Get the percentage of each color in the image
    total_pixels = len(pixels)
    percentages = [(pixel_counts[i] / total_pixels) * 100 for i in range(k)]
    
    # Convert BGR colors to hexadecimal format
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(color[2]), int(color[1]), int(color[0])) for color in colors]
    
    # Combine hex colors with their percentages
    color_percentage_pairs = list(zip(hex_colors, percentages))
    
    # Sort by percentage in descending order
    sorted_color_percentage_pairs = sorted(color_percentage_pairs, key=lambda x: x[1], reverse=True)
    
    return sorted_color_percentage_pairs

# Example usage
if __name__ == "__main__":
    # Load image using OpenCV (change path to your image)
    image = cv2.imread('img1.jpg')
    
    # Extract colors and their percentages
    k = 5  # You can adjust the number of colors/clusters as needed
    sorted_colors = extract_exact_colors(image, k)
    
    # Display results
    for color, percentage in sorted_colors:
        print(f"Color: {color}, Percentage: {percentage:.2f}%")
