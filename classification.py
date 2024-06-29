from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
import torch


# Initialize the model and processor
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# Define categories
categories = [
    "graffiti",
    "garbage",
    "broken_window",
    "green_spaces",
    "public_buildings",
    "sports_and_social_events"
]

# Directory containing images
image_dir = '/Users/kailashkumar/Documents/broken_window/downloaded_files'

# Load and classify images
results = {}
for image_name in os.listdir(image_dir):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):  # check for image files
        path = os.path.join(image_dir, image_name)
        image = Image.open(path)

        # Preprocess image and prepare text inputs
        inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)

        # Model prediction
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image -> text similarity score
        probs = logits_per_image.softmax(dim=1)  # we apply softmax to normalize the scores

        # Get the highest probability category
        best_category = categories[probs.argmax().item()]

        # Store result
        results[image_name] = best_category

# Print classification results
for image_name, category in results.items():
    print(f"{image_name}: {category}")
