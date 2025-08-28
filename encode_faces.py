import face_recognition
import os
import pickle

# Path to dataset
DATASET_PATH = "dataset"
ENCODINGS_FILE = "encodings.pickle"

# Initialize lists
known_encodings = []
known_names = []
known_roll_numbers = []

# Load images and extract encodings
for filename in os.listdir(DATASET_PATH):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        try:
            # Extract roll number and name from filename (e.g., "101_John_Doe.jpg")
            roll_no, name = filename.split("_", 1)
            name = os.path.splitext(name)[0]  # Remove file extension

            # Load and encode face
            image_path = os.path.join(DATASET_PATH, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])  # Store the first face encoding
                known_names.append(name)  # Store the name
                known_roll_numbers.append(roll_no)  # Store roll number
            else:
                print(f"Warning: No faces found in {filename}")

        except ValueError:
            print(f"Skipping {filename}: Incorrect filename format. Use 'RollNo_Name.jpg'")

# Save encodings with roll numbers
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names, "roll_numbers": known_roll_numbers}, f)

print("Face encodings saved successfully.")
