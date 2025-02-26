from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Define class labels
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Define image transformations (must match training preprocessing)
transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load the trained model
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=False)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(class_labels))  # 6 classes

    def forward(self, xb):
        return self.network(xb)

# Load the model and set it to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet().to(device)
model.load_state_dict(torch.load("best.pt", map_location=device))  # Load weights
model.eval()

# Prediction function
def predict_image(image):
    image = transformations(image).unsqueeze(0).to(device)  # Convert to batch format
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, dim=1)
    return class_labels[predicted.item()]  # Return class name

# Flask route to handle image uploads
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))

    # Make prediction
    predicted_class = predict_image(image)

    return jsonify({"predicted_class": predicted_class})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
