# from flask_cors import CORS
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision.models import resnet50, ResNet50_Weights
# from flask import Flask, request, jsonify
# from PIL import Image
# import io

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Define class labels
# class_labels = ['glass', 'metal', 'paper', 'plastic']

# # Define image transformations (must match training preprocessing)
# transformations = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

# # Load the trained model
# class ResNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.network = resnet50(weights=ResNet50_Weights.DEFAULT)  # ✅ Fixed torchvision warning
#         num_ftrs = self.network.fc.in_features
#         self.network.fc = nn.Linear(num_ftrs, len(class_labels))  # 6 classes

#     def forward(self, xb):
#         return self.network(xb)

# # Load the model and set it to evaluation mode
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ResNet().to(device)
# model.load_state_dict(torch.load("v9.pt", map_location=device))  # Load weights
# model.eval()

# # Prediction function
# def predict_image(image):
#     try:
#         image = transformations(image).unsqueeze(0).to(device)  # Convert to batch format
#         with torch.no_grad():
#             outputs = model(image)
#             _, predicted = torch.max(outputs, dim=1)
#         return class_labels[predicted.item()]  # Return class name
#     except Exception as e:
#         return str(e)  # Return error message

# # Flask route to check API status
# @app.route("/", methods=["GET"])
# def home():
#     return "Flask API is running!"

# # Flask route to handle image uploads
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
    
#     try:
#         image = Image.open(io.BytesIO(file.read()))
#         predicted_class = predict_image(image)
#         return jsonify({"predicted_class": predicted_class})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Run the Flask app
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=False)  # ✅ Debug is OFF for production


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
model.load_state_dict(torch.load("v2.pt", map_location=device))
model.eval()

# 🔹 Prediction function with confidence
def predict_image(image):
    image = transformations(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        confidence_value = confidence.item()
        predicted_label = class_labels[predicted.item()]
        if confidence_value < 0.80:
            return "unknown", confidence_value
        return predicted_label, confidence_value

# 🔹 Save image function (before and after background removal)
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
        
        # Save the image after background removal
        removed_bg_image_path = save_image(image_no_bg, image_type="removed")
    except Exception as e:
        return jsonify({"error": f"Background removal failed: {str(e)}"}), 500

    # Predict the class of the image after background removal
    predicted_class, confidence = predict_image(image_no_bg)

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "original_image": original_image_path,
        "removed_bg_image": removed_bg_image_path
    })

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
