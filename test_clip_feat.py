import torch
import clip
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Load and preprocess an image
image = Image.open("WechatIMG105.jpg")
image = preprocess(image).unsqueeze(0).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image)
    # Step0: input image size: 224*224
    # x: torch.Size([1, 3, 224, 224])
    # Step1: change to patch: [*, width, grid, grid], here width is the number of filters in conv1
    # conv1(x): : torch.Size([1, 1024, 16, 16])
    # Step2: flatten: [*, width, grid*grid]
    # reshape: torch.Size([1, 1024, 256])
    # Step3: transpose: [*, grid*grid, width]
    # transpose: torch.Size([1, 256, 1024])
    # Step4: add class token: [*, grid*grid+1, width]
    # class_token: torch.Size([1, 257, 1024])
    # Step5: add position embedding: [*, grid*grid+1, width]
    # pos_embedding: torch.Size([1, 257, 1024])



# image_features now contains the feature vector for the image
