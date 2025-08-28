import face_recognition
import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os
import time

ENCODINGS_FILE = "encodings.pickle"
ATTENDANCE_FILE = "attendance.csv"

# Load known face encodings
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]
known_roll_numbers = data["roll_numbers"]

# Create or load attendance file
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Roll No", "Name", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)
else:
    df = pd.read_csv(ATTENDANCE_FILE)

def mark_attendance(roll_no, name):
    """Mark attendance for a recognized person"""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Load attendance CSV and ensure correct columns
    try:
        attendance_df = pd.read_csv(ATTENDANCE_FILE)
    except Exception:
        attendance_df = pd.DataFrame(columns=["Roll No", "Name", "Time"])

    attendance_df.columns = attendance_df.columns.str.strip()  # Remove spaces in column names

    # Check if the person is already marked today
    if "Roll No" in attendance_df.columns and "Time" in attendance_df.columns:
        if not ((attendance_df["Roll No"].astype(str) == str(roll_no)) & 
                (attendance_df["Time"].str.startswith(now.strftime("%Y-%m-%d")))).any():
            new_entry = pd.DataFrame([[roll_no, name, timestamp]], columns=["Roll No", "Name", "Time"])
            attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
            attendance_df.to_csv(ATTENDANCE_FILE, index=False)
            print(f"Marked attendance for {roll_no} - {name}")
        else:
            print(f"{name} (Roll No: {roll_no}) is already marked today.")
    else:
        print("CSV file columns are incorrect. Creating a new attendance file.")
        attendance_df = pd.DataFrame(columns=["Roll No", "Name", "Time"])
        attendance_df.to_csv(ATTENDANCE_FILE, index=False)
def mark_absentees():
    """Mark absent students who are not in today's attendance"""
    attendance_df = pd.read_csv(ATTENDANCE_FILE)

    today = datetime.now().strftime("%Y-%m-%d")

    # Create a list of roll numbers who are present
    present_students = attendance_df[attendance_df["Time"].str.startswith(today)]["Roll No"].astype(str).tolist()

    # Identify absent students
    absent_students = [(roll_no, name) for roll_no, name in zip(known_roll_numbers, known_names) 
                       if str(roll_no) not in present_students]

    # Add absent students to the record
    for roll_no, name in absent_students:
        new_entry = pd.DataFrame([[roll_no, name, today + " 00:00:00", "Absent"]], 
                                 columns=["Roll No", "Name", "Time", "Status"])
        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)

    attendance_df.to_csv(ATTENDANCE_FILE, index=False)

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = np.argmin(face_distances) if matches else -1

        if best_match_index != -1:
            name = known_names[best_match_index]
            roll_no = known_roll_numbers[best_match_index]
            mark_attendance(roll_no, name)
        else:
            name, roll_no = "Unknown", "N/A"

        # Draw rectangle and text
        top, right, bottom, left = location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{roll_no} - {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    

start_time = time.time()
timeout_duration = 10  # seconds

while True:
    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > timeout_duration:
        break

    

video_capture.release()
cv2.destroyAllWindows()

mark_absentees()