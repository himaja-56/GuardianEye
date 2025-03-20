import cv2
import numpy as np
import speech_recognition as sr
import threading
from flask import Flask, Response, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Face Detection and Gender Model
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Mean values & Labels
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

# Load Models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# **Global Counters** - Updated in real time
current_males = 0
current_females = 0
latest_speech = ""
latest_alert = False

# **Distress Words**
DANGER_WORDS = {
    "help", "save me", "stop", "leave me alone", "don't touch me", "police",
    "scared", "go away", "no", "he's following me", "he's harassing me", "i'm being attacked", "danger"
}

import time  # Import time to track last female voice detected time

# Global variables
latest_danger_alert = False  # New variable to track the alert state
female_voice_detected_time = None  # Tracks last female voice detected time

def detect_faces():
    """Detect faces and update gender count in real time."""
    global current_males, current_females, latest_danger_alert, female_voice_detected_time
    cap = cv2.VideoCapture(0)
    padding = 20

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frameHeight, frameWidth = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # Reset counts for the current frame
        males_in_frame = 0
        females_in_frame = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)

                face = frame[max(0, y1 - padding):min(y2 + padding, frameHeight - 1),
                             max(0, x1 - padding):min(x2 + padding, frameWidth - 1)]

                if face.shape[0] < 50 or face.shape[1] < 50:
                    continue

                # Process face for gender prediction
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                gender = genderList[genderNet.forward()[0].argmax()]

                # Update counts
                if gender == "Female":
                    females_in_frame += 1
                else:
                    males_in_frame += 1

                # Draw bounding box and label
                color = (0, 255, 0) if gender == "Female" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Update global counters
        current_males = males_in_frame
        current_females = females_in_frame

        # 🚨 **Trigger danger condition if males >= 3 and females == 1**
        if males_in_frame >= 3 and females_in_frame == 1:
            if female_voice_detected_time and time.time() - female_voice_detected_time > 10:
                latest_danger_alert = True
            else:
                latest_danger_alert = "Waiting for female voice..."

        else:
            latest_danger_alert = False  # Reset alert if the condition is not met

        # Encode frame to JPEG format
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    """Stream real-time video"""
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts')
def get_counts():
    """Returns the current number of males and females detected in the frame, along with any alerts."""
    return jsonify({
        "males": current_males,
        "females": current_females,
        "danger_alert": latest_danger_alert  # Add the alert message
    })

@app.route('/get_speech_status')
def get_speech_status():
    """Returns the latest detected speech and alert status"""
    return jsonify({"transcript": latest_speech, "alert": latest_alert})

@app.route('/detect_speech')
def detect_speech():
    """Detect speech for distress words and update the alert state."""
    global latest_speech, latest_alert, female_voice_detected_time
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🔊 Adjusting for ambient noise... (stay silent)")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        print("🎤 Listening for distress words...")
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)
            text = recognizer.recognize_google(audio).lower()
            print(f"📝 Detected Speech: {text}")

            is_danger = any(word in text for word in DANGER_WORDS)
            latest_speech = text
            latest_alert = is_danger

            # 🚨 If a female voice is detected, reset the timer
            if "i" in text or "am" in text or "me" in text:
                female_voice_detected_time = time.time()
                print("🎤 Female voice detected. Resetting danger timer.")

            return jsonify({"transcript": text, "alert": is_danger})

        except sr.WaitTimeoutError:
            return jsonify({"transcript": "No speech detected.", "alert": False})
        except sr.UnknownValueError:
            return jsonify({"transcript": "Could not understand audio.", "alert": False})
        except sr.RequestError:
            return jsonify({"transcript": "API unavailable.", "alert": False})


# Run speech recognition in a separate thread
speech_thread = threading.Thread(target=detect_speech, daemon=True)
speech_thread.start()

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
