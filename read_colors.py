import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from typing import List, Tuple

def apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to smooth out minor variations like shadows."""
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

def resize_image(image: np.ndarray, width: int = 300) -> np.ndarray:
    """Resize image to speed up processing."""
    h, w, _ = image.shape
    if w > width:
        height = int((width / w) * h)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image

def extract_colors_until_threshold(image: np.ndarray, min_percentage: float = 1.0, max_colors: int = 10) -> List[Tuple[str, float]]:
    """Dynamically extract colors until percentage contribution is below the threshold."""
    
    pixels = image.reshape((-1, 3))
    total_pixels = len(pixels)
    
    # Start with a few clusters and increase the clusters dynamically
    k = 2
    final_colors = []
    
    # We will keep increasing clusters until either we reach max_colors or the additional colors contribute below min_percentage
    while k <= max_colors:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048)
        kmeans.fit(pixels)
        
        # Get the colors (centroids) from the clusters
        colors = kmeans.cluster_centers_.astype(int)
        
        # Count the number of pixels assigned to each cluster
        pixel_counts = Counter(kmeans.labels_)
        
        # Get the percentage of each color in the image
        percentages = [(pixel_counts[i] / total_pixels) * 100 for i in range(k)]
        
        # Convert BGR colors to hexadecimal format
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(color[2]), int(color[1]), int(color[0])) for color in colors]
        
        # Combine hex colors with their percentages
        color_percentage_pairs = list(zip(hex_colors, percentages))
        
        # Sort by percentage in descending order
        sorted_color_percentage_pairs = sorted(color_percentage_pairs, key=lambda x: x[1], reverse=True)
        
        # Filter colors that meet the minimum percentage
        final_colors = [pair for pair in sorted_color_percentage_pairs if pair[1] >= min_percentage]
        
        # Stop once the number of final colors reaches the desired max_colors or if remaining colors are insignificant
        if len(final_colors) >= max_colors:
            break
        
        # Increase k and retry to capture more significant colors
        k += 1
    
    return final_colors

# Main function to apply everything
def process_image(image_path: str, min_percentage: float = 1.0, max_colors: int = 10, resize_width: int = 300) -> List[Tuple[str, float]]:
    """Load an image, apply preprocessing, and extract significant colors dynamically based on percentage."""
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize image to reduce processing time
    image_resized = resize_image(image, width=resize_width)
    
    # Apply Gaussian blur to reduce shadows
    image_blurred = apply_gaussian_blur(image_resized)
    
    # Extract dynamic colors
    dominant_colors = extract_colors_until_threshold(image_blurred, min_percentage=min_percentage, max_colors=max_colors)
    
    return dominant_colors

# Example usage
if __name__ == "__main__":
    # Load image and process it
    image_path = 'img3.jpg'  # Update this to the path of your image
    min_percentage = 1.0  # Minimum percentage threshold for colors
    max_colors = 15  # Maximum number of significant colors to extract
    sorted_colors = process_image(image_path, min_percentage=min_percentage, max_colors=max_colors)
    
    # Display results
    for color, percentage in sorted_colors:
        print(f"Color: {color}, Percentage: {percentage:.2f}%")
