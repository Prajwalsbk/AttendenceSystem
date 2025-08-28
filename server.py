from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

DATASET_PATH = "dataset"
ATTENDANCE_FILE = "attendance.csv"

# Load face encodings dynamically from dataset folder
def load_encodings():
    known_encodings = []
    known_names = []
    known_roll_numbers = []

    for filename in os.listdir(DATASET_PATH):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            try:
                roll_no, name = filename.split("_", 1)
                name = os.path.splitext(name)[0]  # Remove file extension

                image_path = os.path.join(DATASET_PATH, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(name)
                    known_roll_numbers.append(roll_no)
                else:
                    print(f"⚠ Warning: No face found in {filename}")
            except ValueError:
                print(f"⚠ Skipping {filename}: Incorrect filename format. Use 'RollNo_Name.jpg'")

    return known_encodings, known_names, known_roll_numbers

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        roll_no = request.form['roll_no']
        name = request.form['name']
        file = request.files['photo']
        if file:
            filename = f"{roll_no}_{name}.jpg"
            file.save(os.path.join(DATASET_PATH, filename))
        return redirect(url_for('admin'))
    return render_template('admin.html')

@app.route('/attendance')
def attendance():
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Roll No", "Name", "Time"])
    return render_template('attendance.html', records=df.to_dict(orient='records'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    video_capture = cv2.VideoCapture(0)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Load encodings dynamically before every frame processing
        known_encodings, known_names, known_roll_numbers = load_encodings()

        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            name, roll_no = "Unknown", "N/A"

            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_encodings, encoding))
                name = known_names[best_match_index]
                roll_no = known_roll_numbers[best_match_index]
                mark_attendance(roll_no, name)

            top, right, bottom, left = location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{roll_no} - {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

def mark_attendance(roll_no, name):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame(columns=["Roll No", "Name", "Time"])
    
    # Check if already marked today
    if not ((df["Roll No"].astype(str) == str(roll_no)) & (df["Time"].str.startswith(now.strftime("%Y-%m-%d")))).any():
        df = pd.concat([df, pd.DataFrame([[roll_no, name, timestamp]], columns=["Roll No", "Name", "Time"])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

if __name__ == '__main__':
    app.run(debug=True)
