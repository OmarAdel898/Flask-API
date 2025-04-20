# from flask_cors import CORS
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# from flask import Flask, request, jsonify
# from PIL import Image
# import requests
# import io
# import os
# from datetime import datetime

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Class labels
# class_labels = ['glass', 'metal', 'paper', 'plastic', 'trash']

# # Image preprocessing
# transformations = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

# # Model definition
# class ResNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.network = models.resnet50(pretrained=False)
#         num_ftrs = self.network.fc.in_features
#         self.network.fc = nn.Linear(num_ftrs, len(class_labels))

#     def forward(self, xb):
#         return self.network(xb)

# # Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ResNet().to(device)
# model.load_state_dict(torch.load("v6.pt", map_location=device))
# model.eval()

# # 🔹 Save image to Desktop with dynamic filename
# def save_image_to_desktop(image, prefix):
#     desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(desktop_path, f"{prefix}_{timestamp}.png")
#     image.save(filename)
#     return filename

# # 🔹 Remove background using Remove.bg API
# def remove_background(image_bytes):
#     # api_key = "FBqHa2pbZ1tRp7Np1zFgcudm"
#     api_key="3Cp6VDzwYemYM77wKa8UpPoX"
#     response = requests.post(
#         'https://api.remove.bg/v1.0/removebg',
#         files={'image_file': image_bytes},
#         data={'size': 'auto'},
#         headers={'X-Api-Key': api_key},
#     )

#     if response.status_code == requests.codes.ok:
#         return Image.open(io.BytesIO(response.content)).convert("RGB")
#     else:
#         print("Remove.bg Error:", response.status_code, response.text)
#         return None

# # 🔹 Predict class
# def predict_image(image):
#     image = transformations(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, dim=1)
#     return class_labels[predicted.item()]

# @app.route("/", methods=["GET"])
# def home():
#     return "Flask API is running!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     image_bytes = file.read()
#     original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#     # Save original image to Desktop
#     save_image_to_desktop(original_image, "original")

#     # Remove background
#     image_no_bg = remove_background(io.BytesIO(image_bytes))
#     if image_no_bg is None:
#         return jsonify({"error": "Background removal failed"}), 500

#     # Predict class
#     predicted_class = predict_image(image_no_bg)

#     # Save background-removed image with predicted class name
#     save_image_to_desktop(image_no_bg, predicted_class)

#     return jsonify({"predicted_class": predicted_class})

# # Run the app
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)



# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# from PIL import Image
# import io
# from rembg import remove

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Define class labels
# class_labels = ['glass', 'metal', 'paper', 'plastic', 'trash']

# # Define image transformations
# transformations = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

# # Define your model architecture
# class ResNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.network = models.resnet50(pretrained=False)
#         num_ftrs = self.network.fc.in_features
#         self.network.fc = nn.Linear(num_ftrs, len(class_labels))

#     def forward(self, xb):
#         return self.network(xb)

# # Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ResNet().to(device)
# model.load_state_dict(torch.load("best5.pt", map_location=device))
# model.eval()

# # Prediction function
# def predict_image(image):
#     image = transformations(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, dim=1)
#     return class_labels[predicted.item()]

# # Routes
# @app.route("/", methods=["GET"])
# def home():
#     return "Flask API is running!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     input_bytes = file.read()

#     # Remove background using rembg
#     try:
#         output_data = remove(input_bytes)
#         image = Image.open(io.BytesIO(output_data)).convert("RGB")  # Ensure 3-channel input
#     except Exception as e:
#         return jsonify({"error": f"Background removal failed: {str(e)}"}), 500

#     # Predict the class
#     predicted_class = predict_image(image)
#     return jsonify({"predicted_class": predicted_class})

# # Run the app
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)



from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from rembg import remove
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define class labels
class_labels = ['glass', 'metal', 'paper', 'plastic', 'trash']

# Define image transformations
transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Define your model architecture
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=False)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(class_labels))

    def forward(self, xb):
        return self.network(xb)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet().to(device)
model.load_state_dict(torch.load("v7.pt", map_location=device))
model.eval()

# Prediction function
def predict_image(image):
    image = transformations(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, dim=1)
    return class_labels[predicted.item()]

# Save image function (before and after background removal)
def save_image(image, image_type="before"):
    folder_path = "saved_images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate file name based on image type
    file_name = f"{image_type}_background_removed_image.png"
    image_path = os.path.join(folder_path, file_name)
    
    # Save the image
    image.save(image_path)
    return image_path

# Routes
@app.route("/", methods=["GET"])
def home():
    return "Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    input_bytes = file.read()

    # Save the original image before background removal
    try:
        input_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
        original_image_path = save_image(input_image, image_type="original")
    except Exception as e:
        return jsonify({"error": f"Failed to save original image: {str(e)}"}), 500

    # Remove background using rembg
    try:
        output_data = remove(input_bytes)
        image_no_bg = Image.open(io.BytesIO(output_data)).convert("RGB")
        # Predict the class of the image after background removal
        predicted_class = predict_image(image_no_bg)
        # Save the image after background removal
        removed_bg_image_path = save_image(image_no_bg, image_type=predicted_class)
    except Exception as e:
        return jsonify({"error": f"Background removal failed: {str(e)}"}), 500

    

    return jsonify({
        "predicted_class": predicted_class,
        "original_image": original_image_path,  # Path to the saved original image
        "removed_bg_image": removed_bg_image_path  # Path to the saved processed image
    })

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
