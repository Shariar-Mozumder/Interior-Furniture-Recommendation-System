import numpy as np
import pandas as pd
from colorsys import rgb_to_hsv, hsv_to_rgb

# Helper function to convert HEX to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

# Helper function to convert RGB to HEX
def rgb_to_hex(rgb_color):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

# Helper function to convert RGB to HSV
def rgb_to_custom_hsv(rgb_color):
    return rgb_to_hsv(*rgb_color)

# Helper function to convert HSV to RGB
def hsv_to_custom_rgb(hsv_color):
    return hsv_to_rgb(*hsv_color)

# Function to extract features from a list of HSV colors
def extract_features_from_hsv(hsv_colors):
    features = {}
    
    # Extracting Hue (C1)
    features['chsv-D1-C1'] = hsv_colors[0][0]  # Hue of the 1st color
    features['chsv-D2-C1'] = hsv_colors[1][0]  # Hue of the 2nd color
    features['chsv-D3-C1'] = hsv_colors[2][0]  # Hue of the 3rd color
    
    # Extracting Saturation (C2)
    features['chsv-D1-C2'] = hsv_colors[0][1]  # Saturation of the 1st color
    features['chsv-D2-C2'] = hsv_colors[1][1]  # Saturation of the 2nd color
    features['chsv-D3-C2'] = hsv_colors[2][1]  # Saturation of the 3rd color
    
    # Extracting Value (C3)
    features['chsv-D1-C3'] = hsv_colors[0][2]  # Value of the 1st color
    features['chsv-D2-C3'] = hsv_colors[1][2]  # Value of the 2nd color
    features['chsv-D3-C3'] = hsv_colors[2][2]  # Value of the 3rd color

    return features

# Function to calculate compatibility score using Lasso weights
def calculate_compatibility(features, model_weights):
    scores = {}
    for feature, value in features.items():
        if feature in model_weights.index:
            scores[feature] = value * model_weights.loc[feature].sum()  # Sum the weights for each feature across models
    return scores

# Function to generate a compatible color in HSV space
def generate_compatible_color(compatibility_score, base_hsv_color):
    # Modify the hue of the base color based on the compatibility score
    new_hue = (base_hsv_color[0] + compatibility_score) % 1.0  # Keep hue between 0 and 1
    return (new_hue, base_hsv_color[1], base_hsv_color[2])

# Main function to find a compatible color based on 3 input hex colors
def find_compatible_color(hex_colors, weights_csv_path):
    # Convert hex colors to RGB and then to HSV
    hsv_colors = [rgb_to_custom_hsv(hex_to_rgb(color)) for color in hex_colors]
    
    # Extract features from the input colors
    features = extract_features_from_hsv(hsv_colors)
    
    # Load the weights from the CSV file
    weights_df = pd.read_csv(weights_csv_path)
    weights_df.set_index('Feature', inplace=True)
    
    # Calculate the compatibility scores using the weights for each color
    compatibility_scores = calculate_compatibility(features, weights_df)
    
    # Generate a new compatible color (using the first color as a base)
    total_compatibility_score = sum(compatibility_scores.values())
    compatible_hsv_color = generate_compatible_color(total_compatibility_score, hsv_colors[0])
    
    # Convert the new HSV color back to RGB and then to Hex
    compatible_rgb_color = hsv_to_custom_rgb(compatible_hsv_color)
    compatible_hex_color = rgb_to_hex(compatible_rgb_color)
    
    return compatible_hex_color, compatibility_scores

# Example usage: Input 3 hex colors and path to CSV file
input_colors = ['#252525', '#908d88', '#c7d9dc']  # Example: orange, green, and blue
csv_file_path = 'weights.csv'  # Replace with actual path to your weights file

# Find a compatible color and get the scores for each input color
compatible_color, scores = find_compatible_color(input_colors, csv_file_path)

# Output the compatible color and the individual scores for each color
print(f"Compatible color for {input_colors} is: {compatible_color}")
print("Compatibility scores for each feature:")
for feature, score in scores.items():
    print(f"{feature}: {score}")
