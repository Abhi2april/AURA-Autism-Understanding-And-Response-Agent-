import cv2
import numpy as np
import math
import pyaudio
import wave
import threading
import time
import os
from collections import Counter
from deepface import DeepFace
import requests
import json
import tempfile

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize current emotion (will be updated when faces are detected)
current_emotion = 'neutral'
emotion_detected = False
last_printed_emotion = None
transcript_buffer = []

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5  # Record in 5-second chunks

# Groq API settings
GROQ_API_KEY = "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"  # Replace with your actual Groq API key
WHISPER_MODEL = "whisper-large-v3-turbo"
GROQ_API_URL = f"https://api.groq.com/openai/v1/audio/transcriptions"

def draw_emotion_clock(frame, emotion):
    """Draw circular emotion clock interface with indicator pointing to detected emotion"""
    height, width, _ = frame.shape

    # Set up dimensions for the emotion clock
    clock_center = (width - 150, 150)
    clock_radius = 100
    indicator_length = 80

    # Draw background circle
    cv2.circle(frame, clock_center, clock_radius, (0, 0, 0), -1)
    cv2.circle(frame, clock_center, clock_radius, (255, 255, 255), 2)

    # Find the emotion index
    try:
        max_emotion_idx = emotion_labels.index(emotion.lower())
    except ValueError:
        # If emotion not in our list, default to neutral
        max_emotion_idx = emotion_labels.index('neutral')

    # Draw emotion labels around the clock
    for i, emotion_label in enumerate(emotion_labels):
        angle = i * 2 * math.pi / len(emotion_labels)
        label_x = int(clock_center[0] + (clock_radius + 20) * math.sin(angle))
        label_y = int(clock_center[1] - (clock_radius + 20) * math.cos(angle))

        # Highlight the detected emotion
        if i == max_emotion_idx:
            cv2.putText(frame, emotion_label.capitalize(), (label_x - 30, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, emotion_label.capitalize(), (label_x - 30, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Draw indicator (arrow pointing to the detected emotion)
    angle = max_emotion_idx * 2 * math.pi / len(emotion_labels)
    indicator_x = int(clock_center[0] + indicator_length * math.sin(angle))
    indicator_y = int(clock_center[1] - indicator_length * math.cos(angle))

    # Draw arrow
    cv2.line(frame, clock_center, (indicator_x, indicator_y), (0, 255, 0), 3)

    # Draw arrow head
    arrow_head_length = 15
    arrow_angle1 = angle - math.pi/8
    arrow_angle2 = angle + math.pi/8
    arrow_x1 = int(indicator_x - arrow_head_length * math.sin(arrow_angle1))
    arrow_y1 = int(indicator_y + arrow_head_length * math.cos(arrow_angle1))
    arrow_x2 = int(indicator_x - arrow_head_length * math.sin(arrow_angle2))
    arrow_y2 = int(indicator_y + arrow_head_length * math.cos(arrow_angle2))

    cv2.line(frame, (indicator_x, indicator_y), (arrow_x1, arrow_y1), (0, 255, 0), 3)
    cv2.line(frame, (indicator_x, indicator_y), (arrow_x2, arrow_y2), (0, 255, 0), 3)

    # Display current emotion
    cv2.putText(frame, f"Emotion: {emotion.capitalize()}",
                (width - 300, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display transcript with emotions
    display_transcript(frame)

def display_transcript(frame):
    """Display the transcript with emotion tags at the bottom of the frame"""
    height, width, _ = frame.shape

    # Create a semi-transparent overlay for the transcript area
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 150), (width, height), (0, 0, 0), -1)
    alpha = 0.7  # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Display the transcript
    y_pos = height - 120
    font_size = 0.6
    line_height = 30

    if not transcript_buffer:
        cv2.putText(frame, "Waiting for speech...", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)
        return

    # Display the most recent entries (last 4 lines)
    display_lines = transcript_buffer[-4:]
    for line in display_lines:
        cv2.putText(frame, line, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)
        y_pos += line_height

def record_audio():
    """Record audio for transcription"""
    global current_emotion, last_printed_emotion, transcript_buffer

    audio = pyaudio.PyAudio()

    # Open audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("* Recording audio...")

    while True:
        frames = []
        # Record for RECORD_SECONDS
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        # Save as WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_filename = temp_audio_file.name

            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

        # Transcribe using Groq API
        transcript = transcribe_audio(temp_filename)

        # Clean up temporary file
        try:
            os.remove(temp_filename)
        except:
            pass

        # If we got a transcript and it's not empty, add to buffer with emotion
        if transcript and transcript.strip():
            # If emotion has changed since last transcript segment or this is the first segment
            if last_printed_emotion != current_emotion or last_printed_emotion is None:
                transcript_line = f"[{current_emotion}] {transcript}"
                last_printed_emotion = current_emotion
            else:
                # Continuation of the same emotion
                transcript_line = transcript

            transcript_buffer.append(transcript_line)
            print(transcript_line)

            # Limit buffer size to keep the most recent entries
            if len(transcript_buffer) > 10:
                transcript_buffer.pop(0)

def transcribe_audio(audio_file_path):
    """Send audio file to Groq API for transcription"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': audio_file}
            data = {
                'model': WHISPER_MODEL,
                'language': 'en'  # You can change this for other languages
            }

            response = requests.post(GROQ_API_URL, headers=headers, files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                return result.get('text', '')
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return ""
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return ""

# Main function
def main():
    global current_emotion, emotion_detected

    # Start the audio recording and transcription in a separate thread
    audio_thread = threading.Thread(target=record_audio)
    audio_thread.daemon = True
    audio_thread.start()

    # Start capturing video
    cap = cv2.VideoCapture(0)

    # For smoother display, we'll add a slight delay before analyzing each frame
    frame_count = 0
    analyze_every = 10  # Analyze every 10 frames

    # For smoothing emotion transitions
    emotion_buffer = []
    buffer_size = 5

    print("Press 'q' to quit")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Mirror the frame horizontally for more intuitive self-viewing
        frame = cv2.flip(frame, 1)

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Only analyze emotions periodically to improve performance
        if frame_count % analyze_every == 0 and len(faces) > 0:
            # For simplicity, just use the first face detected
            x, y, w, h = faces[0]

            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y+h, x:x+w].copy()

            try:
                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Determine the dominant emotion
                new_emotion = result[0]['dominant_emotion']

                # Add to emotion buffer for smoothing
                emotion_buffer.append(new_emotion)
                if len(emotion_buffer) > buffer_size:
                    emotion_buffer.pop(0)

                # Use the most common emotion in the buffer
                emotion_counts = Counter(emotion_buffer)
                current_emotion = emotion_counts.most_common(1)[0][0]

                emotion_detected = True

            except Exception as e:
                print(f"Error in emotion analysis: {str(e)}")
                # Keep using the previous emotion if analysis fails

        # Draw rectangles for all detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        if emotion_detected:
            draw_emotion_clock(frame, current_emotion)
        else:
            draw_emotion_clock(frame, 'neutral')

        cv2.imshow('Emotion Detection with Speech', frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

