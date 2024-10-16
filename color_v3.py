import numpy as np
import pandas as pd
from colorsys import rgb_to_hsv, hsv_to_rgb

# Helper function to convert HEX to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

# Helper function to convert RGB to HEX
def rgb_to_hex(rgb_color):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

# Function to extract features from a list of HSV colors
def extract_features_from_hsv(hsv_colors):
    features = {
        'chsv-D1-C1': hsv_colors[0][0],  # Hue of the 1st color
        'chsv-D2-C1': hsv_colors[1][0],  # Hue of the 2nd color
        'chsv-D3-C1': hsv_colors[2][0],  # Hue of the 3rd color
        'chsv-D1-C2': hsv_colors[0][1],  # Saturation of the 1st color
        'chsv-D2-C2': hsv_colors[1][1],  # Saturation of the 2nd color
        'chsv-D3-C2': hsv_colors[2][1],  # Saturation of the 3rd color
        'chsv-D1-C3': hsv_colors[0][2],  # Value of the 1st color
        'chsv-D2-C3': hsv_colors[1][2],  # Value of the 2nd color
        'chsv-D3-C3': hsv_colors[2][2]   # Value of the 3rd color
    }
    return features

# Function to calculate compatibility score for a single color using Lasso weights
def calculate_color_score(features, model_weights):
    score = 0.0
    for feature, value in features.items():
        if feature in model_weights.index:
            score += value * model_weights.loc[feature].values[0]  # Access the weight correctly
    return score

# Function to generate variations of colors in the HSV space
def generate_color_variations(base_colors, num_variations=10):
    variations = []
    base_hsv = [rgb_to_hsv(*hex_to_rgb(color)) for color in base_colors]
    
    for _ in range(num_variations):
        # Create variations by adjusting hue, saturation, and value slightly
        for h, s, v in base_hsv:
            # Random adjustments
            new_h = (h + np.random.uniform(-0.1, 0.1)) % 1.0
            new_s = min(max(s + np.random.uniform(-0.1, 0.1), 0), 1)
            new_v = min(max(v + np.random.uniform(-0.1, 0.1), 0), 1)
            variations.append(hsv_to_rgb(new_h, new_s, new_v))
    
    return [rgb_to_hex(rgb) for rgb in variations]

# Function to find compatible colors using weights
def find_compatible_colors(hex_colors, weights_csv_path, num_compatible_colors=3):
    # Load the weights from the CSV file
    weights_df = pd.read_csv(weights_csv_path)
    weights_df.set_index('Feature', inplace=True)

    # Generate color variations
    color_variations = generate_color_variations(hex_colors, num_variations=50)

    # Calculate scores for all generated colors
    scores = []
    unique_colors = set()  # To avoid duplicates
    for color in color_variations:
        rgb = hex_to_rgb(color)
        hsv = rgb_to_hsv(*rgb)
        
        # Extract features for the generated color
        new_features = extract_features_from_hsv([hsv] * 3)  # Use the same color for features
        score = calculate_color_score(new_features, weights_df)
        
        # Store unique colors with their scores
        if color not in unique_colors:
            unique_colors.add(color)
            scores.append((color, score))

    # Sort colors by their scores
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select the top compatible colors
    top_colors = [color for color, score in scores[:num_compatible_colors]]
    top_scores = [score for color, score in scores[:num_compatible_colors]]

    return top_colors, top_scores

# Example usage: Input 3 hex colors and path to CSV file
input_colors = ['#8f8d87', '#c7d9dc', '#252424']  # Example colors
csv_file_path = 'weights.csv'  # Replace with actual path to your weights file

# Find compatible colors (at least 3) and get the scores for each color
compatible_colors, scores = find_compatible_colors(input_colors, csv_file_path, num_compatible_colors=3)

# Output the compatible colors and their scores
print(f"Compatible colors for {input_colors} are: {compatible_colors}")
print("Scores for each compatible color:")
for color, score in zip(compatible_colors, scores):
    print(f"{color}: {score:.4f}")
