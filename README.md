# hand-sign-to-text-convertor
Real-time deef and dumb hand sign recognition using Python, OpenCV, MediaPipe and MLP

Real-Time Hand Sign Recognition Using Machine Learning

📌 Project Overview

This project is a real-time hand sign recognition system developed using Python, OpenCV, MediaPipe, and Machine Learning. It recognizes static hand gestures representing English alphabets (A–Z), SPACE, and DELETE using a webcam.

The system extracts 21 hand landmarks using MediaPipe Hands, processes the landmark coordinates, and classifies gestures using a Multi-Layer Perceptron (MLP) classifier. Recognized characters are combined to form sentences, and the output can be converted into speech using an offline Text-to-Speech engine.

🎯 Objectives

- Recognize static hand gestures in real time using a webcam.
- Identify English alphabets from A to Z.
- Support SPACE and DELETE gestures for sentence formation.
- Apply machine learning to classify hand gestures.
- Convert recognized text into speech without requiring an online TTS service.
- Provide an accessible and user-friendly human-computer interaction system.

🛠️ Technologies Used

Category| Technology
Programming Language| Python 3.x
Computer Vision| OpenCV
Hand Landmark Detection| MediaPipe Hands
Machine Learning| Scikit-learn MLPClassifier
Data Processing| NumPy, Pandas
Feature Extraction| MediaPipe Hand Landmarks
Feature Scaling| StandardScaler
Model Storage| Pickle
Text-to-Speech| pyttsx3
Development Environment| Visual Studio Code
Version Control| Git and GitHub

🧠 Machine Learning Algorithm

Multi-Layer Perceptron (MLP) Classifier

The project uses a Multi-Layer Perceptron (MLP), a feed-forward artificial neural network, to classify hand gestures from extracted landmark features.

Model configuration:

- Hidden layers: "(256, 128)" or "(128, 64)", depending on the implemented version.
- Activation function: ReLU.
- Optimizer: Adam.
- Early stopping: Enabled if configured in the training code.
- Maximum iterations: As specified in the model configuration.

The model predicts one of 28 gesture classes: A–Z, SPACE, or DELETE.

✋ Hand Landmark Detection and Feature Extraction

MediaPipe Hands detects 21 landmarks on a hand.

Each landmark contains three coordinates:

- X: Horizontal coordinate.
- Y: Vertical coordinate.
- Z: Relative depth coordinate.

Therefore, the feature vector contains:

21 landmarks × 3 coordinates = 63 features

The extracted coordinates are normalized relative to the wrist and scaled using a reference distance, such as the wrist-to-middle-finger distance. This helps reduce variations caused by hand position and gesture size.

📊 Dataset

The dataset contains 28 gesture classes:

- Alphabets: A–Z (26 classes)
- SPACE: Space gesture
- DEL: Delete gesture

The images are organized into class-specific folders. MediaPipe is used to extract landmark features from the images, and the resulting numerical features can be stored in a CSV file for model training.

Dataset size, image count, and the source of the dataset can be added after verification.

⚙️ Project Workflow

Webcam Input
     ↓
OpenCV Video Capture
     ↓
MediaPipe Hand Detection
     ↓
21 Hand Landmark Extraction
     ↓
63 Coordinate Features
     ↓
Landmark Normalization
     ↓
Feature Scaling
     ↓
MLP Classifier
     ↓
Prediction Confidence Check
     ↓
Stable Prediction
     ↓
Sentence Formation
     ↓
Text-to-Speech Output

🔍 Data Preprocessing

1. Load the hand gesture images.
2. Convert image color format from BGR to RGB.
3. Detect hand landmarks using MediaPipe.
4. Extract X, Y, and Z coordinates.
5. Normalize the landmark coordinates.
6. Generate numerical feature vectors.
7. Store the extracted features and class labels in a CSV file.
8. Apply feature scaling before model training and prediction.

📈 Model Evaluation

The model can be evaluated using the following techniques:

- Stratified K-Fold Cross-Validation.
- Accuracy score.
- Confusion matrix.
- Classification report.
- Precision, recall, and F1-score.

Actual evaluation results should be reported after running the model on the dataset. Cross-validation and test evaluation must be performed without data leakage.

🚀 Key Features

- Real-time hand detection through a webcam.
- Recognition of 26 English alphabets.
- SPACE and DELETE gesture support.
- Sentence formation from recognized characters.
- Confidence-based prediction filtering.
- Stable prediction and cooldown timing.
- Offline text-to-speech output.
- Snapshot capture functionality.

📦 Installation

1. Clone the Repository

git clone https://github.com/pragavans98/hand-sign-to-text-convertor.git



2. Create a Virtual Environment

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

3. Install Dependencies

pip install opencv-python mediapipe numpy pandas scikit-learn pyttsx3

Use package versions compatible with your Python version and the code in your project.

▶️ How to Run

1. Clone or download the repository.
2. Install the required dependencies.
3. Ensure the trained model and any required feature scaler files are available.
4. Connect a webcam.
5. Run the project's main Python script.

python main.py

Replace "main.py" with the actual filename of your application entry point.

📁 Suggested Project Structure

Hand-Sign-Recognition/
│
├── dataset/
│   ├── A/
│   ├── B/
│   ├── C/
│   ├── ...
│   ├── Z/
│   ├── SPACE/
│   └── DEL/
│
├── features/
│   └── landmarks.csv
│
├── models/
│   └── hand_sign_model.pkl
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

This is a suggested structure. Adjust the filenames and folders to match your actual project.

🔮 Future Enhancements

- Support for dynamic gestures and continuous sign sequences.
- Recognition of complete words and phrases.
- Improved performance under different lighting conditions.
- Larger and more diverse training datasets.
- Evaluation across different users and hand positions.
- A graphical user interface for easier interaction.

🎓 Applications

- Assistive human-computer interaction.
- Educational sign-language demonstrations.
- Gesture-based input systems.
- Accessibility-focused software prototypes.

⚠️ Limitations

- The current approach focuses on static hand gestures.
- Recognition accuracy depends on dataset quality and gesture similarity.
- Webcam lighting, hand orientation, and occlusion may affect predictions.
- Recognizing individual alphabets does not provide full sign-language translation, which requires additional linguistic and temporal information.

confusion matrix:


Cross-validation accuracy: 0.8749529190207156
Confusion Matrix:
 [[286   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0  89   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0  26   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0  71   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0  37   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0 225   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0  56   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0  92   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  77   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   1   0   0   0 155   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0 100   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0 155   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0 164   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0  50   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  59   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  83   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  72   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 113
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   52   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0
    0  73   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0  31   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0  53   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0  58   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0  64   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0  16   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0  28   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0 277   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0  90]]
Classification Report:
               precision    recall  f1-score   support

           A       1.00      1.00      1.00       287
           B       1.00      1.00      1.00        89
           C       1.00      1.00      1.00        26
           D       1.00      1.00      1.00        71
         DEL       1.00      1.00      1.00        37
           E       1.00      1.00      1.00       225
           F       1.00      1.00      1.00        56
           G       1.00      1.00      1.00        92
           H       1.00      1.00      1.00        77
           I       1.00      0.99      1.00       156
           J       1.00      1.00      1.00       100
           K       1.00      1.00      1.00       155
           L       1.00      1.00      1.00       164
           M       0.96      1.00      0.98        50
           N       1.00      1.00      1.00        59
           O       1.00      1.00      1.00        83
           P       1.00      1.00      1.00        72
           Q       1.00      1.00      1.00       113
           R       1.00      1.00      1.00        52
           S       1.00      0.99      0.99        74
       SPACE       1.00      1.00      1.00        31
           T       1.00      1.00      1.00        53
           U       1.00      1.00      1.00        58
           V       1.00      1.00      1.00        64
           W       1.00      1.00      1.00        16
           X       1.00      1.00      1.00        28
           Y       1.00      1.00      1.00       277
           Z       1.00      1.00      1.00        90

    accuracy                           1.00      2655
   macro avg       1.00      1.00      1.00      2655
weighted avg       1.00      1.00      1.00      2655

MLP model trained and saved to models/sign_model_mlp.pkl
👨‍💻 Author

Project: Real-Time Hand Sign Recognition Using Machine Learning

Developed using: Python, OpenCV, MediaPipe, and Scikit-learn.

📄 License

Add a suitable open-source license if you intend to distribute this project publicly. Ensure that your dataset's license permits redistribution before uploading it.
