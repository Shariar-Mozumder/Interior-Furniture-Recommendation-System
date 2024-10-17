# import cv2
# import numpy as np
# import torch

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can choose different models like 'yolov5m' or 'yolov5l'

# def get_color(image, mask):
#     """Get the dominant color within the masked area."""
#     masked_image = cv2.bitwise_and(image, image, mask=mask)
#     pixels = masked_image[mask > 0]
#     if len(pixels) == 0:
#         return None
#     average_color = np.mean(pixels, axis=0).astype(int)
#     return '#{:02x}{:02x}{:02x}'.format(average_color[2], average_color[1], average_color[0])  # BGR to HEX

# def label_detections(image, results):
#     """Label detected objects with their respective colors."""
#     labeled_image = image.copy()
#     for i in range(len(results.xyxy[0])):  # Iterate over the detections
#         xyxy = results.xyxy[0][i]  # Get the bounding box
#         conf = results.xyxy[0][i][4]  # Get confidence score
#         cls = int(results.xyxy[0][i][5])  # Get class index

#         x1, y1, x2, y2 = map(int, xyxy[:4])  # Extract coordinates
#         label = f"{model.names[cls]} {conf:.2f}"
        
#         # Create a mask for the detected object
#         mask = np.zeros(image.shape[:2], dtype=np.uint8)
#         mask[y1:y2, x1:x2] = 255  # Simple rectangular mask
        
#         # Get the color of the detected object
#         dominant_color = get_color(image, mask)
        
#         # Draw bounding box and label
#         cv2.rectangle(labeled_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(labeled_image, f"{label}: {dominant_color}", (x1, y1 - 5), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     return labeled_image

# def process_image(image_path: str) -> np.ndarray:
#     """Load an image, apply preprocessing, and detect objects."""
#     # Load the image
#     image = cv2.imread(image_path)
    
#     # Resize image for processing
#     image_resized = cv2.resize(image, (640, 640))  # Resize to match YOLOv5 input size
    
#     # Perform object detection
#     results = model(image_resized)

#     # Label detections
#     labeled_image = label_detections(image_resized, results)
    
#     return labeled_image

# # Example usage
# if __name__ == "__main__":
#     # Load image and process it
#     image_path = 'img3.jpg'  # Update this to the path of your image
#     labeled_image = process_image(image_path)
    
#     # Display results
#     cv2.imshow('Labeled Detections', labeled_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import torch
# from torchvision import transforms

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can choose different models like 'yolov5m' or 'yolov5l'

# # Load your segmentation model
# # Using DeepLabV3 with ResNet backbone for segmentation
# segmentation_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# segmentation_model.eval()  # Set the model to evaluation mode

# def get_color(image, mask):
#     """Get the dominant color within the masked area."""
#     masked_image = cv2.bitwise_and(image, image, mask=mask)
#     pixels = masked_image[mask > 0]
    
#     if len(pixels) == 0:
#         return '#808080'  # Return gray if no pixels are found

#     average_color = np.mean(pixels, axis=0).astype(int)
#     return '#{:02x}{:02x}{:02x}'.format(average_color[2], average_color[1], average_color[0])  # BGR to HEX

# def segment_image(image):
#     """Segment the image into walls, ceilings, and floors."""
#     preprocess = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((512, 512)),  # Resize for the segmentation model
#         transforms.ToTensor(),
#     ])

#     input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():
#         output = segmentation_model(input_tensor)
    
#     # Get the segmentation map
#     segmentation_map = output['out'].argmax(dim=1).squeeze().cpu().numpy()

#     return segmentation_map

# def label_detections(image, results, segmentation_map):
#     """Label detected objects with their respective colors and identify walls, ceilings, and floors."""
#     labeled_image = image.copy()
#     height, width = segmentation_map.shape

#     # Create masks for walls, ceilings, and floors
#     wall_mask = (segmentation_map == 0).astype(np.uint8) * 255  # Assuming 0 = wall
#     ceiling_mask = (segmentation_map == 1).astype(np.uint8) * 255  # Assuming 1 = ceiling
#     floor_mask = (segmentation_map == 2).astype(np.uint8) * 255  # Assuming 2 = floor

#     # Ensure that the masks are the same size as the image
#     wall_mask = cv2.resize(wall_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
#     ceiling_mask = cv2.resize(ceiling_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
#     floor_mask = cv2.resize(floor_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

#     # Get colors for each segmented area
#     wall_color = get_color(image, wall_mask)
#     ceiling_color = get_color(image, ceiling_mask)
#     floor_color = get_color(image, floor_mask)

#     # Draw the segmented areas
#     cv2.putText(labeled_image, f"Wall Color: {wall_color}", (10, 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     cv2.putText(labeled_image, f"Ceiling Color: {ceiling_color}", (10, 50), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     cv2.putText(labeled_image, f"Floor Color: {floor_color}", (10, 70), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     for i in range(len(results.xyxy[0])):  # Iterate over the detections
#         xyxy = results.xyxy[0][i]  # Get the bounding box
#         conf = results.xyxy[0][i][4]  # Get confidence score
#         cls = int(results.xyxy[0][i][5])  # Get class index

#         x1, y1, x2, y2 = map(int, xyxy[:4])  # Extract coordinates
#         label = f"{model.names[cls]} {conf:.2f}"

#         # Create a mask for the detected object
#         object_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create a black mask
#         object_mask[y1:y2, x1:x2] = 255  # Fill the bounding box with white

#         # Get color for the detected object
#         object_color = get_color(image, object_mask)

#         # Draw bounding box and label with color
#         cv2.rectangle(labeled_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(labeled_image, f"{label} Color: {object_color}", (x1, y1 - 5), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     return labeled_image


# def process_image(image_path: str) -> np.ndarray:
#     """Load an image, apply preprocessing, and detect objects."""
#     # Load the image
#     image = cv2.imread(image_path)
    
#     # Resize image for processing
#     image_resized = cv2.resize(image, (640, 640))  # Resize to match YOLOv5 input size
    
#     # Perform object detection
#     results = model(image_resized)

#     # Segment the image
#     segmentation_map = segment_image(image_resized)

#     # Label detections
#     labeled_image = label_detections(image_resized, results, segmentation_map)
    
#     return labeled_image

# # Example usage
# if __name__ == "__main__":
#     # Load image and process it
#     image_path = 'wall.jpg'  # Update this to the path of your image
#     labeled_image = process_image(image_path)
    
#     # Display results
#     cv2.imshow('Labeled Detections', labeled_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()




import cv2
import numpy as np
import torch
from torchvision import transforms

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load DeepLabV3 model for segmentation
segmentation_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
segmentation_model.eval()  # Set the model to evaluation mode

def segment_image(image):
    """Segment the image into walls, ceilings, and floors."""
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),  # Resize for the segmentation model
        transforms.ToTensor(),
    ])

    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = segmentation_model(input_tensor)['out']
    
    # Get the segmentation map
    segmentation_map = output.argmax(dim=1).squeeze().cpu().numpy()
    
    # Resize segmentation map back to original image size
    segmentation_map = cv2.resize(segmentation_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    return segmentation_map

def get_colors(image, mask, num_colors=5):
    """Get the dominant colors within the masked area."""
    # Ensure the mask is of type uint8
    mask = mask.astype(np.uint8)

    # Masked image to get colors
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    pixels = masked_image[mask > 0]
    
    if len(pixels) == 0:
        return ['#808080']  # Return gray if no pixels are found

    # Get colors using KMeans clustering
    pixels = pixels.reshape(-1, 3)
    if len(pixels) > 0:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(c[2], c[1], c[0]) for c in colors]
        return hex_colors
    return ['#808080']  # Default if no pixels


def label_detections(image, results, segmentation_map):
    """Label detected objects and surrounding colors."""
    labeled_image = image.copy()
    height, width = segmentation_map.shape

    # Create masks for walls, ceilings, and floors
    wall_mask = (segmentation_map == 0).astype(np.uint8) * 255  # Assuming 0 = wall
    ceiling_mask = (segmentation_map == 1).astype(np.uint8) * 255  # Assuming 1 = ceiling
    floor_mask = (segmentation_map == 2).astype(np.uint8) * 255  # Assuming 2 = floor

    # Get colors for each segmented area
    wall_colors = get_colors(image, wall_mask)
    ceiling_colors = get_colors(image, ceiling_mask)
    floor_colors = get_colors(image, floor_mask)

    # Draw the colors on the labeled image
    cv2.putText(labeled_image, f"Wall Colors: {', '.join(wall_colors)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(labeled_image, f"Ceiling Colors: {', '.join(ceiling_colors)}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(labeled_image, f"Floor Colors: {', '.join(floor_colors)}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Print the colors for walls, ceilings, and floors
    print(f"Wall Colors: {', '.join(wall_colors)}")
    print(f"Ceiling Colors: {', '.join(ceiling_colors)}")
    print(f"Floor Colors: {', '.join(floor_colors)}")

    for i in range(len(results.xyxy[0])):  # Iterate over the detections
        xyxy = results.xyxy[0][i]  # Get the bounding box
        conf = results.xyxy[0][i][4]  # Get confidence score
        cls = int(results.xyxy[0][i][5])  # Get class index

        x1, y1, x2, y2 = map(int, xyxy[:4])  # Extract coordinates
        label = f"{model.names[cls]} {conf:.2f}"

        # Create a mask for the detected object
        object_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create a black mask
        object_mask[y1:y2, x1:x2] = 255  # Fill the bounding box with white

        # Get color for the detected object
        object_color = get_colors(image, object_mask)

        # Print the detected object label and color
        print(f"Detected: {label}, Color: {object_color}")

        # Draw bounding box and label with color
        cv2.rectangle(labeled_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(labeled_image, f"{label} Color: {', '.join(object_color)}", (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return labeled_image

def process_image(image_path: str) -> np.ndarray:
    """Load an image, apply preprocessing, and detect objects."""
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize image for processing
    image_resized = cv2.resize(image, (640, 640))  # Resize to match YOLOv5 input size
    
    # Perform object detection
    results = model(image_resized)

    # Segment the image
    segmentation_map = segment_image(image_resized)

    # Label detections
    labeled_image = label_detections(image_resized, results, segmentation_map)
    
    return labeled_image

# Example usage
if __name__ == "__main__":
    # Load image and process it
    image_path = 'wall.jpg'  # Update this to the path of your image
    labeled_image = process_image(image_path)
    
    # Display results
    cv2.imshow('Labeled Detections', labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

