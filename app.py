import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
import joblib
from sklearn.preprocessing import MinMaxScaler
# Image transformation for preprocessing
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from flask import Flask, render_template, request, redirect, url_for , send_file , Response ,jsonify
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import io
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from skimage import color, filters, measure
from skimage import io as skio, color, filters, measure, morphology, segmentation
import matplotlib
import pickle
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import cv2
import pydicom




# Initialize Flask app
app = Flask(__name__)



upload_folder = os.path.join(f"uploads")
os.makedirs(upload_folder, exist_ok=True)


app.config.update(SECRET_KEY="rms-brain")





login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"{self.id}"


# Sample hardcoded user data (username and plain text password)
users = {
    "sanaat": "sanaat_123123"  # Plain text password
}


# Load user from the ID
@login_manager.user_loader
def load_user(userid):
    return User(userid)


@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.id)


@app.route("/login")
def login():
    return render_template("sign-in.html")


@app.route("/login", methods=["POST"])
def loggin():
    username = request.form["username"]
    password = request.form["password"]

    # Check if the username exists and password matches in plain text
    if username in users and users[username] == password:
        user = User(username)
        login_user(user)  # Log the user in
        return redirect(url_for("dashboard"))  # Redirect to the protected dashboard
    else:
        return render_template("sign-in.html", error="Username or password is invalid")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# Error handler for unauthorized access
@app.errorhandler(401)
def page_not_found(e):
    return Response("""
                    <html><center style='background-color:white;'>
                    <h2 style='color:red;'>Login failed</h2>
                    <h1>Error 401</h1>
                    </center></html>""")





## MAMOGRAPHY



# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






# Define the Flask route to render the classifier page
@app.route("/brain_stroke_classifier", methods=["GET"])
def brain_stroke_classifier_get():
    return render_template("brain_stroke_classifier.html")



# Define the model class for brain stroke detection
class SimpleCNN_brain_stroke(nn.Module):
    def __init__(self):
        super(SimpleCNN_brain_stroke, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 2)  # Output for 2 classes: Normal, Stroke

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)  # Flatten the output
        x = self.fc1(x)
        return x

# Initialize the model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
brain_stroke_model = SimpleCNN_brain_stroke().to(device)

# Load the model weights (ensure that 'brain_stroke_classifier.pth' exists)
brain_stroke_model.load_state_dict(torch.load('brain_stroke_classifier.pth', map_location=device))

# Set the model to evaluation mode
brain_stroke_model.eval()

# Define class names for prediction
brain_stroke_class_names = ['Normal', 'Stroke']

# Define image transformations (same as in the model training)
brain_stroke_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Helper function to process the uploaded image
def process_image_brain_stroke(image):
    try:
        # Convert image to RGB (in case it's not)
        img = image.convert('RGB')
        img = brain_stroke_transform(img)  # Apply transformations
        img = img.unsqueeze(0)  # Add batch dimension (1 batch)
        return img.to(device)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")



# Define Flask route to handle image upload and prediction
@app.route("/predict_brain_stroke", methods=["POST"])
def predict_brain_stroke():    
    file = request.files['stroke_image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image file
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({'error': f"Error loading image: {str(e)}"}), 400
    
    try:
        # Process and predict the image
        processed_image = process_image_brain_stroke(image)
        
        # Predict using the model
        with torch.no_grad():
            outputs = brain_stroke_model(processed_image)
            _, predicted = torch.max(outputs, 1)
        
        # Map prediction to class name
        predicted_class = brain_stroke_class_names[predicted.item()]
        
        # Optionally, compute confidence score
        softmax = torch.nn.Softmax(dim=1)
        confidence = softmax(outputs).max().item()

        return render_template( "brain_stroke_classifier.html" , prediction = predicted_class)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500




## Multiple Sclerosis

# Define Flask route to render the classifier page
@app.route("/ms_classifier", methods=["GET"])
def ms_classifier_get():
    return render_template("ms_classifier.html")


# Define the SimpleCNN model for MS classification
class SimpleCNN_ms(nn.Module):
    def __init__(self):
        super(SimpleCNN_ms, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 128)  # Match saved model's dimensions
        self.fc2 = nn.Linear(128, 4)  # Add this layer for the output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)  # Flatten the output
        x = F.relu(self.fc1(x))  # Pass through fc1 with activation
        x = self.fc2(x)  # Output layer
        return x

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ms = SimpleCNN_ms().to(device)

# Load the state_dict (model weights) from the file
model_ms.load_state_dict(torch.load('ms_classifier.pth', map_location=device))

# Set the model to evaluation mode
model_ms.eval()

# Define class names for prediction
ms_classes = ['Control-Axial', 'Control-Sagittal', 'MS-Axial', 'MS-Sagittal']

# Define image transformations (same as in the model training)
ms_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Helper function to process the uploaded image
def process_image_ms(image):
    try:
        # Convert image to RGB (in case it's not)
        img = image.convert('RGB')
        img = ms_transform(img)  # Apply transformations
        img = img.unsqueeze(0)  # Add batch dimension (1 batch)
        return img.to(device)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")



# Define Flask route to handle image upload and prediction
@app.route("/ms_classifier", methods=["POST"])
def predict_ms_classifier():
    file = request.files['ms_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image file
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({'error': f"Error loading image: {str(e)}"}), 400
    
    try:
        # Process and predict the image
        processed_image = process_image_ms(image)
        
        # Predict using the model
        with torch.no_grad():
            outputs = model_ms(processed_image)
            _, predicted = torch.max(outputs, 1)
        
        # Map prediction to class name
        predicted_class = ms_classes[predicted.item()]
        
        # Optionally, compute confidence score
        softmax = torch.nn.Softmax(dim=1)
        confidence = softmax(outputs).max().item()

        return render_template("ms_classifier.html"  , prediction =  predicted_class)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500






## BRAINTUMOR 


@app.route('/brain_tumor_classifier')
def brain_tumor_classifier_get():
    return render_template("brain_tumor_classifier.html")


# Define the CNN model for brain tumor classification
class SimpleCNN_brain(nn.Module):
    def __init__(self):
        super(SimpleCNN_brain, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 4)  # Output for 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)  # Flatten the output
        x = self.fc1(x)
        return x

# Initialize the model and load pretrained weights
model_brain = SimpleCNN_brain().to(device)
model_brain.load_state_dict(torch.load('monai_brain_classifier.pth', map_location=device))
model_brain.eval()

# Image preprocessing function for brain tumor classification
def preprocess_image_brain(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    # Resize image to match the model's input size
    img = cv2.resize(img, (224, 224))
    
    # Convert to RGB, normalize, and convert to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = transform(img)
    img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
    return img

# Prediction function for brain tumor classification
def predict_tumor_brain(image_path):
    processed_image = preprocess_image_brain(image_path)
    with torch.no_grad():
        outputs = model_brain(processed_image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
    
    # Define the classes
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    predicted_label = classes[predicted_class]
    
    # Optionally, compute confidence score using softmax
    softmax = torch.nn.Softmax(dim=1)
    confidence = softmax(outputs).max().item()
    
    return predicted_label, confidence

# Route to handle image upload and prediction
@app.route('/brain_tumor_classifier', methods=['POST'])
def upload_and_predict_brain_tumor():    
    file = request.files['braintumor_image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Secure the file name and save it
    filename = file.filename
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    
    try:
        # Make the prediction using the pre-trained model
        predicted_class, confidence = predict_tumor_brain(file_path)
        return render_template( "brain_tumor_classifier.html" , prediction = predicted_class)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500






@app.route('/parkinson_detection')
def parkinson_detection_get():
    return render_template('parkinson_detection.html')

## PARKINSON 
# Load the trained model and scaler
clf = joblib.load('parkinsons_model.pkl')
scaler = joblib.load('parkinson_scaler.pkl')

# Define the route for the home page


# Define the route for the form submission
@app.route('/parkinson_prediction', methods=['POST'])
def parkinson_prediction_post():
    try:
        # Get the features from the form (as float)
        features = [
            float(request.form['MDVP_Fo']),
            float(request.form['MDVP_Fhi']),
            float(request.form['MDVP_Flo']),
            float(request.form['MDVP_Jitter_percent']),
            float(request.form['MDVP_Jitter_Abs']),
            float(request.form['MDVP_RAP']),
            float(request.form['MDVP_PPQ']),
            float(request.form['Jitter_DDP']),
            float(request.form['MDVP_Shim']),
            float(request.form['MDVP_Shim_dB']),
            float(request.form['Shimmer_APQ3']),
            float(request.form['Shimmer_APQ5']),
            float(request.form['MDVP_APQ']),
            float(request.form['Shimmer_DDA']),
            float(request.form['NHR']),
            float(request.form['HNR']),
            float(request.form['RPDE']),
            float(request.form['DFA']),
            float(request.form['spread1']),
            float(request.form['spread2']),
            float(request.form['D2']),
            float(request.form['PPE'])
        ]
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Make the prediction
        prediction = clf.predict(features_scaled)
        
        # Prepare the result to be passed to the template
        result = "Parkinson's Detected" if prediction == 1 else "No Parkinson's Detected"
        
        # Render the result template with the prediction
        return render_template('parkinson_detection.html', prediction=result)

    except Exception as e:
        return f"Error occurred: {str(e)}"



@app.route("/dicom_viewer")
def dicom_viewer_get():
    return render_template("dicom_viewer.html")


@app.route("/dicomtools/multi", methods=["POST"])
@login_required
def dicom_viewer_post():
    try:
        
        # Retrieve DICOM file from request
        dicom_file = request.files.get("dicom")
        
        # Check if a file was uploaded
        if not dicom_file or dicom_file.filename == '':
            return """
                <html>
                    <body style='background-color:black;'>
                        <center>
                            <h2 style='color:red;'>
                                !!! No Selected File !!!
                            </h2>
                            <h1>Please check the information sent from the form and then try again</h1>
                            <a href='/'>
                                <button style='background-color: red; border: none;color: white;padding: 15px 32px;text-align: center;text-decoration: none;display: inline-block;font-size: 16px;'> Return  </button>
                            </a>
                        </center>
                    </body>
                </html>
            """
        
        # Save DICOM file and insert information into the database
        name = dicom_file.filename
        goal_path = os.path.join(upload_folder, name)
        dicom_file.save(goal_path)
        

        
        # Read DICOM file
        ds = pydicom.dcmread(goal_path, force=True)
        # Extract image data
        image = ds.pixel_array
        
        # Get selected colormaps
        colormaps = request.form.get("colormaps", "gray nipy_spectral inferno coolwarm gnuplot jet terrain tab20c").split()
        
        # Determine number of colormaps
        num_colormaps = len(colormaps)
        # Calculate number of rows and columns for subplots
        num_rows = (num_colormaps - 1) // 4 + 1  # Maximum 4 colormaps per row
        num_cols = min(num_colormaps, 4)
        # Create subplots for each colormap
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        # Plot image with each colormap
        for i, cmap in enumerate(colormaps):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.imshow(image, cmap=cmap)
            ax.set_title(cmap)
            ax.axis('off')
        # Hide empty subplots
        for i in range(num_colormaps, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            axes[row, col].axis('off')
        plt.tight_layout()

        # Save plot to a bytes buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()

        # Return the image as a file attachment
        return send_file(img_buffer, mimetype='image/png')
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        app.logger.error(error_message)
        return error_message, 500






@app.route("/about_us")
def about_us():
    return render_template("profile.html")


@app.route("/tables")
def table():
    return render_template("tables.html")


# Run the app
if __name__ == '__main__':
    app.run(port = 5005 , debug = True)
    
    
