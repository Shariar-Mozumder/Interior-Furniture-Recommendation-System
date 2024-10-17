import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# Load a pre-trained model without loading a .pth file
model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=8, activation='softmax')
model.eval()  # Set the model to evaluation mode

# Load and preprocess your image
image_path = 'img3.jpg'  # Replace with your image path
image = Image.open(image_path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image
    transforms.ToTensor(),  # Convert to tensor
])
image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Perform segmentation
with torch.no_grad():
    output = model(image_tensor)
    segmentation = output.argmax(dim=1).squeeze().numpy()

# Visualize results
plt.imshow(segmentation)
plt.axis('off')
plt.show()
