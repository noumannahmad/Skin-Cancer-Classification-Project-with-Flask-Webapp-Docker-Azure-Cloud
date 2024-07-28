from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import load_model
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Define the number of classes (benign and malignant)
num_classes = 2
class_names = ['benign', 'malignant']

# Load the PyTorch model
model_path = './resnet50_model.pth.tar'
model = load_model(model_path, num_classes)
model.eval()  # Set model to evaluation mode


# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def is_jpg_file(filename):
    return filename.lower().endswith('.jpg')

def evaluate(net, image):
    with torch.no_grad():
        output = net(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

def predict_value(image_path):
    try:
        image = preprocess_image(image_path)
        prediction = evaluate(model, image)
        return class_names[prediction]
    except Exception as e:
        print("Error:", str(e))
        return str(e)

@app.route("/", methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            if is_jpg_file(file.filename):
                img_path = os.path.join("uploads", file.filename)
                if not os.path.exists("uploads"):
                    os.makedirs("uploads")
                file.save(img_path)
                prediction = predict_value(img_path)
                return render_template("index.html", prediction=prediction)
            else:
                return "Please upload a JPG file."
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and is_jpg_file(file.filename):
            img_path = os.path.join("uploads", file.filename)
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            file.save(img_path)
            prediction = predict_value(img_path)
            return prediction
        else:
            return "Please upload a JPG file."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
