# Skin Cancer Classification Project

## Overview

This project implements a skin cancer classification application using deep learning. It employs a ResNet-50 model trained to classify images of skin lesions as either "Benign" or "Malignant." The application is built with Flask for serving predictions and Docker for containerization. The final Docker container is deployed on Azure Cloud Platform.

## Dataset: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign?resource=download

## Project Components

1. **Model Training**: Training a ResNet-50 model to classify skin lesions.
2. **Flask Application**: Serving the model predictions via a web interface.
3. **Docker Container**: Packaging the application for deployment.
4. **Deployment**: Deploying the Docker container to Azure Cloud.

## Getting Started

### Prerequisites

- Docker
- Python 3.9+
- Gunicorn
- Flask

### Directory Structure

```
/project-root
|-- app.py
|-- model.py
|-- requirements.txt
|-- resnet50_model.pth.tar
|-- static
|   |-- css
|       |-- main.css
|   |-- js
|       |-- main.js
|-- templates
|   |-- base.html
|   |-- index.html
|-- uploads
|-- Dockerfile

```

### Training the Model

1. **Prepare the Dataset**: Organize your dataset into folders for training and validation, with subfolders for each class (e.g., `Benign` and `Malignant`).

2. **Training Script** (`train_model.py`):

   ```python
   import torch
   import torchvision.models as models
   import torch.nn as nn
   from torchvision import transforms, datasets
   from torch.utils.data import DataLoader

   def train_model(model, criterion, optimizer, dataloaders, num_epochs=25):
       for epoch in range(num_epochs):
           model.train()
           running_loss = 0.0
           running_corrects = 0
           for inputs, labels in dataloaders['train']:
               optimizer.zero_grad()
               outputs = model(inputs)
               loss = criterion(outputs, labels)
               loss.backward()
               optimizer.step()
               running_loss += loss.item() * inputs.size(0)
               _, preds = torch.max(outputs, 1)
               running_corrects += torch.sum(preds == labels.data)
           epoch_loss = running_loss / len(dataloaders['train'].dataset)
           epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
           print(f'Epoch {epoch}/{num_epochs - 1}')
           print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
       return model

   def main():
       # Define model, criterion, optimizer, and dataloaders
       model = models.resnet50(pretrained=True)
       num_ftrs = model.fc.in_features
       model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes: Benign and Malignant
       criterion = nn.CrossEntropyLoss()
       optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
       # Define your dataloaders
       dataloaders = {'train': DataLoader(datasets.ImageFolder('train', transforms.Compose([...]), batch_size=4, shuffle=True))}
       model = train_model(model, criterion, optimizer, dataloaders)
       torch.save(model.state_dict(), 'resnet50_model.pth')

   if __name__ == '__main__':
       main()
   ```

### Flask Application

1. **Create Flask App** (`app.py`):

   ```python
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

   ```

2. **Model Loading** (`model.py`):

   ```python
    import torch
    import torchvision.models as models
    import torch.nn as nn

    def load_model(model_path, num_classes):
        model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        return model_ft

   ```

### Docker Container

1. **Create Dockerfile**:

   ```dockerfile
    FROM python:3.9-slim

    # Set the working directory
    WORKDIR /app

    # Copy the requirements file into the container
    COPY requirements.txt requirements.txt

    # Install the dependencies
    RUN pip install -r requirements.txt

    # Copy the rest of the application code
    COPY . .

    # Expose the port the app runs on
    EXPOSE 8080

    # Command to run the application using gunicorn
    CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]


   ```

2. **Build and Run Docker Container**:

   ```sh
   docker build -t skin-cancer-webapp .
   docker run -p 8080:8080 skin-cancer-app
   ```

### Deployment to Azure Cloud

1. **Push Docker Image to Dokcer Hub Registry (GCR)**:

   ```sh

   docker tag skin_app_image noumannahmad/flask-webapp-skin-cancer:1.0

   #If your app requires linux/amd64, you need to rebuild the Docker image \for the correct architecture. You can specify the target platform using the --platform flag:

   docker build --platform linux/amd64 -t docker.io/noumannahmad/flask-webapp-skin-cancer:1.0 .
   docker push docker.io/noumannahmad/flask-webapp-skin-cancer:1.0

   ```

2. **Deploy to Azure Cloud Run**:

   ```sh
   Create account, adn deply the docker container 
   ```

### Summary

This README provides a comprehensive guide to:
1. Training a ResNet-50 model for skin cancer classification.
2. Setting up a Flask app to serve the model and provide predictions.
3. Containerizing the app with Docker.
4. Deploying the Docker container to Azure Cloud.

Feel free to modify paths and filenames based on your specific setup and requirements.

