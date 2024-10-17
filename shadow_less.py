import cv2
import numpy as np

def rgb_to_hex(rgb):
    """Convert an RGB color to HEX format."""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def generate_shadowless_image(image: np.ndarray, blur_ksize: int = 5, quantization_level: int = 15) -> np.ndarray:
    """
    Generate a shadow-less, solid-color version of the input image by reducing color variations due to shadows.
    
    Parameters:
        image (np.ndarray): The input image in BGR format.
        blur_ksize (int): Kernel size for Gaussian blur. Larger values blur more, reducing shadow details.
        quantization_level (int): The level of color quantization. Lower values result in fewer unique colors.
        
    Returns:
        np.ndarray: The shadow-less, solid-color image, unique colors, and their counts.
    """
    
    # Step 1: Apply Gaussian Blur to reduce sharp shadow transitions
    blurred_image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    
    # Step 2: Convert image to LAB color space and apply histogram equalization to normalize lighting
    lab_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_channel_eq = cv2.equalizeHist(l_channel)  # Equalize the L (brightness) channel
    lab_image_eq = cv2.merge([l_channel_eq, a_channel, b_channel])
    equalized_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2BGR)
    
    # Step 3: Apply color quantization to merge similar colors
    pixels = equalized_image.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Define criteria for KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Apply KMeans clustering to reduce the number of colors
    _, labels, centers = cv2.kmeans(pixels, quantization_level, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert the centers back to 8-bit values and assign them to the pixels
    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()]
    quantized_image = quantized_image.reshape(equalized_image.shape)
    
    return quantized_image, centers, labels

def label_colors(image: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Label the unique colors in the image with their hex codes and percentages.
    
    Parameters:
        image (np.ndarray): The image to label.
        centers (np.ndarray): The unique colors in the image.
        labels (np.ndarray): The labels from KMeans clustering.
        
    Returns:
        np.ndarray: The image with labeled colors and percentages.
    """
    labeled_image = image.copy()
    unique, counts = np.unique(labels, return_counts=True)
    
    # Total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]
    
    # Prepare color data for labeling
    color_data = []
    for idx, count in zip(unique, counts):
        color_percentage = (count / total_pixels) * 100
        color_data.append((centers[idx], rgb_to_hex(centers[idx]), color_percentage))
    
    # Sort color data by percentage in descending order
    color_data.sort(key=lambda x: x[2], reverse=True)
    
    # Limit to the top 15 colors
    color_data = color_data[:15]

    # Set initial position for labels
    x_offset = 10
    y_offset = 20
    spacing = 40  # Spacing between color labels

    for color, hex_color, percentage in color_data:
        label_position = (x_offset, y_offset)
        
        # Draw rectangle for the color
        cv2.rectangle(labeled_image, (x_offset, y_offset - 20), 
                      (x_offset + 30, y_offset), color.tolist())
        
        # Put the hex color code and percentage as text
        label_text = f"{hex_color} ({percentage:.2f}%)"
        cv2.putText(labeled_image, label_text, 
                    (x_offset + 40, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Update y_offset for the next label
        y_offset += spacing

    return labeled_image

# Example usage
if __name__ == "__main__":
    # Load the image using OpenCV (change to your image path)
    image = cv2.imread('img3.jpg')
    
    # Generate shadow-less, solid color image
    shadowless_image, centers, labels = generate_shadowless_image(image, blur_ksize=5, quantization_level=15)
    
    # Label the colors in the image with hex codes and percentages
    labeled_image = label_colors(shadowless_image, centers, labels)
    
    # Save the result to a file (optional)
    cv2.imwrite('labeled_shadowless_image_with_percentages.jpg', labeled_image)
    
    # Display the result
    cv2.imshow('Labeled Shadow-less Image', labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
