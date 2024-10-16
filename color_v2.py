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

# Helper function to convert RGB to HSV
def rgb_to_custom_hsv(rgb_color):
    return rgb_to_hsv(*rgb_color)

# Helper function to convert HSV to RGB
def hsv_to_custom_rgb(hsv_color):
    return hsv_to_rgb(*hsv_color)

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

# Function to generate compatible colors based on different schemes
def generate_compatible_colors(base_hsv_colors, model_weights, num_colors=3):
    compatible_colors = []
    scores = []

    for i in range(num_colors):
        # Randomly pick a base color from the input colors
        base_hsv_color = base_hsv_colors[np.random.randint(len(base_hsv_colors))]

        # Generate random variations
        hue_variation = np.random.uniform(-0.5, 0.5)  # Larger random variation in hue
        saturation_variation = np.random.uniform(-0.3, 0.3)  # Larger random variation in saturation
        value_variation = np.random.uniform(-0.3, 0.3)  # Larger random variation in value

        # Create a new color based on variations
        new_hue = (base_hsv_color[0] + hue_variation) % 1.0  # Ensure hue is within [0, 1]
        new_saturation = max(0, min(base_hsv_color[1] + saturation_variation, 1))  # Keep saturation in [0, 1]
        new_value = max(0, min(base_hsv_color[2] + value_variation, 1))  # Keep value in [0, 1]
        
        new_color_hsv = (new_hue, new_saturation, new_value)
        new_color_rgb = hsv_to_custom_rgb(new_color_hsv)

        # Extract features for the new color
        new_features = extract_features_from_hsv([new_color_hsv, new_color_hsv, new_color_hsv])
        
        # Calculate the score for the new color
        score = calculate_color_score(new_features, model_weights)

        compatible_colors.append(rgb_to_hex(new_color_rgb))
        scores.append(score)

    # Adding complementary and triadic colors for variety
    for base_hsv_color in base_hsv_colors:
        # Complementary color
        comp_hue = (base_hsv_color[0] + 0.5) % 1.0
        comp_color_hsv = (comp_hue, base_hsv_color[1], base_hsv_color[2])
        comp_color_rgb = hsv_to_custom_rgb(comp_color_hsv)
        comp_features = extract_features_from_hsv([comp_color_hsv, comp_color_hsv, comp_color_hsv])
        comp_score = calculate_color_score(comp_features, model_weights)
        
        compatible_colors.append(rgb_to_hex(comp_color_rgb))
        scores.append(comp_score)

        # Triadic colors
        triadic_colors = [
            ((base_hsv_color[0] + 1/3) % 1.0, base_hsv_color[1], base_hsv_color[2]),
            ((base_hsv_color[0] + 2/3) % 1.0, base_hsv_color[1], base_hsv_color[2])
        ]
        for triadic_hsv in triadic_colors:
            triadic_rgb = hsv_to_custom_rgb(triadic_hsv)
            triadic_features = extract_features_from_hsv([triadic_hsv, triadic_hsv, triadic_hsv])
            triadic_score = calculate_color_score(triadic_features, model_weights)
            
            compatible_colors.append(rgb_to_hex(triadic_rgb))
            scores.append(triadic_score)

    # Filter to get the top num_colors based on scores
    sorted_indices = np.argsort(scores)[::-1][:num_colors]
    return [compatible_colors[i] for i in sorted_indices], [scores[i] for i in sorted_indices]

# Main function to find compatible colors based on 3 input hex colors
def find_compatible_colors(hex_colors, weights_csv_path, num_compatible_colors=3):
    # Convert hex colors to RGB and then to HSV
    hsv_colors = [rgb_to_custom_hsv(hex_to_rgb(color)) for color in hex_colors]

    # Load the weights from the CSV file
    weights_df = pd.read_csv(weights_csv_path)
    weights_df.set_index('Feature', inplace=True)

    # Generate compatible colors using the provided HSV colors
    compatible_colors, scores = generate_compatible_colors(hsv_colors, weights_df, num_colors=num_compatible_colors)
    
    return compatible_colors, scores

# Example usage: Input 3 hex colors and path to CSV file
input_colors = ['#8f8d87', '#c7d9dc', '#252424']  # Example: orange, green, and blue
csv_file_path = 'weights.csv'  # Replace with actual path to your weights file

# Find compatible colors (at least 3) and get the scores for each color
compatible_colors, scores = find_compatible_colors(input_colors, csv_file_path, num_compatible_colors=3)

# Output the compatible colors and their scores
print(f"Compatible colors for {input_colors} are: {compatible_colors}")
print("Scores for each compatible color:")
for color, score in zip(compatible_colors, scores):
    print(f"{color}: {score:.4f}")
