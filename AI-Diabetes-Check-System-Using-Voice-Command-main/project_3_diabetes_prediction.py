# Import necessary libraries
import pandas as pd
import numpy as np
import pyttsx3  # Text-to-Speech conversion library
import speech_recognition as sr  # For voice recognition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to speak out the text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech using the microphone
def recognize_speech(prompt):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(prompt)
        speak(prompt)
        audio = r.listen(source)
        
        try:
            speech_text = r.recognize_google(audio)
            print(f"You said: {speech_text}")
            return speech_text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            speak("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Speech recognition service is unavailable.")
            return None

# Load your dataset (Update with your file path)
data = pd.read_csv("C:/Users/saisa/OneDrive/Desktop/diabetes prediction/AI-Diabetes-Check-System-Using-Voice-Command-main/diabetes_data_upload.csv")

# Display the first few rows to understand the structure
print(data.head())

# Encoding categorical features (e.g., Yes/No questions) using LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder to all columns with 'Yes'/'No'
for column in data.columns:
    if data[column].dtype == 'object' and data[column].nunique() <= 2:  # Likely 'Yes/No' or binary values
        data[column] = le.fit_transform(data[column])

# Assume 'class' is the target and needs to be encoded too
if 'class' in data.columns:
    data['class'] = le.fit_transform(data['class'])

# Split the data into features (X) and target (y)
X = data.drop('class', axis=1)
y = data['class']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the model's accuracy and performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

speak(f'Model training complete with an accuracy of {accuracy:.2f}')

# Asking the user for input (via speech or text)
print("Enter the patient's details to predict diabetes:")
speak("Enter the patient's details to predict diabetes.")

# Collecting the necessary user inputs
def get_input(prompt, use_voice=False):
    if use_voice:
        speech_input = recognize_speech(prompt)
        if speech_input:
            return speech_input.strip().lower()
    return input(prompt).strip().lower()

# Collect inputs via speech or text
use_voice = input("Would you like to provide input via voice? (yes/no): ").strip().lower() == 'yes'

Age = int(get_input("Age: ", use_voice))
Gender = get_input("Gender (Male/Female): ", use_voice)
Polyuria = get_input("Polyuria (Yes/No): ", use_voice)
Polydipsia = get_input("Polydipsia (Yes/No): ", use_voice)
sudden_weight_loss = get_input("Sudden weight loss (Yes/No): ", use_voice)
weakness = get_input("Weakness (Yes/No): ", use_voice)
Polyphagia = get_input("Polyphagia (Yes/No): ", use_voice)
Genital_thrush = get_input("Genital thrush (Yes/No): ", use_voice)
visual_blurring = get_input("Visual blurring (Yes/No): ", use_voice)
Itching = get_input("Itching (Yes/No): ", use_voice)
Irritability = get_input("Irritability (Yes/No): ", use_voice)
delayed_healing = get_input("Delayed healing (Yes/No): ", use_voice)
partial_paresis = get_input("Partial paresis (Yes/No): ", use_voice)
muscle_stiffness = get_input("Muscle stiffness (Yes/No): ", use_voice)
Alopecia = get_input("Alopecia (Yes/No): ", use_voice)
Obesity = get_input("Obesity (Yes/No): ", use_voice)

# Encode the binary inputs using the same logic as used in the dataset
binary_map = {'yes': 1, 'no': 0}
Gender = 1 if Gender.lower() == 'male' else 0  # Assuming 1 for Male and 0 for Female

# Create a numpy array for the new patient data
new_patient = np.array([[Age, Gender, 
                         binary_map[Polyuria], binary_map[Polydipsia], binary_map[sudden_weight_loss], 
                         binary_map[weakness], binary_map[Polyphagia], binary_map[Genital_thrush], 
                         binary_map[visual_blurring], binary_map[Itching], binary_map[Irritability], 
                         binary_map[delayed_healing], binary_map[partial_paresis], 
                         binary_map[muscle_stiffness], binary_map[Alopecia], binary_map[Obesity]]])

# Scale the new patient data
new_patient_scaled = scaler.transform(new_patient)

# Make a prediction
prediction = model.predict(new_patient_scaled)

# Output the prediction result with speech
if prediction[0] == 1:
    result = "The patient is predicted to have diabetes."
else:
    result = "The patient is predicted not to have diabetes."

print(result)
speak(result)
