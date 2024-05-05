import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp


class VideoProcessor:
    """Handles video processing for person tracking and pose estimation using YOLO and MediaPipe."""

    def __init__(self):
        """Initializes the video processor with YOLO model and MediaPipe pose estimation."""
        self.net = cv2.dnn.readNet(r"model/yolov4.weights", r"model/yolov4.cfg")
        self.classes = [line.strip() for line in open(r"model/coco.names", "r")]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cap = None

    def process_frame(self, frame):
        """Processes each frame of the video to detect persons and estimate their poses."""
        if frame is None:
            print("Empty frame received.")
            return None  # Return None if the frame is empty

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and self.classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) == 0:
            return frame  # Return the original frame if no detections

        # Default box color
        box_color = (0, 255, 0)  # Default to green box

        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            if y + h > height or x + w > width:
                continue  # Skip boxes that extend outside the frame

            person_roi = frame[y:y + h, x:x + w]
            if person_roi.size == 0:
                continue  # Skip empty ROIs

            try:
                roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                print(f"Error converting ROI to RGB: {e}")
                continue  # Skip this ROI on error

            results = self.pose.process(roi_rgb)
            if results.pose_landmarks:
                action = self.estimate_action(results.pose_landmarks)
                fall_detected = self.detect_fall(results.pose_landmarks)

                text = action

                if fall_detected:
                    box_color = (0, 0, 255)  # Change to red if fall is detected
                    text = "Falling Down"  # Change text when a fall is detected

                # Calculate the position of the text
                text_location = (x + 5, y + 15) if y > 20 else (x + 5, y + h - 5)

                # Draw the text
                cv2.putText(frame, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        return frame

    def detect_fall(self, pose_landmarks):
        """Detects if a person is falling down based on the position of their nose relative to their hips."""
        landmarks = pose_landmarks.landmark
        nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
        mid_hip = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y +
                   landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y) / 2
        if nose.y > mid_hip - 0.1:
            return True
        return False

    def estimate_action(self, pose_landmarks):
        """Estimates the action being performed by a person based on their pose landmarks."""
        landmarks = pose_landmarks.landmark
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        def calculate_angle(a, b, c):
            """Calculates the angle between three points."""
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            return np.degrees(angle)

        left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        if (abs(l_hip.y - l_ankle.y) < 0.1 or abs(r_hip.y - r_ankle.y) < 0.1) and (
                left_knee_angle < 120 or right_knee_angle < 120):
            return 'Sitting'
        elif left_knee_angle > 165 and right_knee_angle > 165:
            return 'Standing'
        elif (left_knee_angle > 140 or right_knee_angle > 140) and (left_knee_angle < 165 or right_knee_angle < 165):
            return 'Walking'
        else:
            return 'Running'


class VideoTrackerApp:
    """Main application for tracking persons in video streams."""

    def __init__(self, window, window_title):
        """Initializes the main application with a window and title."""
        self.window = window
        self.window.title(window_title)

        self.processor = VideoProcessor()

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack(side=tk.TOP, expand=True)

        self.btn_frame = tk.Frame(window)
        self.btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_open = tk.Button(self.btn_frame, text="Open Video", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT, expand=True)

        self.btn_start = tk.Button(self.btn_frame, text="Start", command=self.start_tracking)
        self.btn_start.pack(side=tk.LEFT, expand=True)

        self.btn_reset = tk.Button(self.btn_frame, text="Reset", command=self.reset_video)
        self.btn_reset.pack(side=tk.LEFT, expand=True)

        self.tracking_active = False  # Flag to check if tracking is already active

    def open_video(self):
        """Opens a file dialog to select and open a video file, resets any ongoing video processing."""
        self.reset_video()
        self.processor.cap = cv2.VideoCapture(filedialog.askopenfilename())
        ret, frame = self.processor.cap.read()
        if ret:
            self.display_image(frame)

    def start_tracking(self):
        """Starts the video tracking process if it's not already running."""
        if not self.processor.cap.isOpened():
            self.processor.cap = cv2.VideoCapture(filedialog.askopenfilename())
        if not self.tracking_active:  # Check if tracking is not already active
            fps = self.processor.cap.get(cv2.CAP_PROP_FPS)
            self.processor.delay = int(1000 / fps)
            self.update_frame()
            self.tracking_active = True  # Set the flag to indicate tracking is active

    def reset_video(self):
        """Resets the video stream by releasing the capture and clearing the canvas, also resets the tracking flag."""
        if self.processor.cap and self.processor.cap.isOpened():
            self.processor.cap.release()
        self.processor.cap = None
        self.canvas.delete("all")
        self.tracking_active = False  # Reset the tracking flag

    def update_frame(self):
        """Updates the frame in the video stream."""
        if self.processor.cap and self.processor.cap.isOpened():
            ret, frame = self.processor.cap.read()
            if ret:
                self.display_image(self.processor.process_frame(frame))
                self.window.after(self.processor.delay, self.update_frame)
            else:
                self.processor.cap.release()
                self.processor.cap = None
                print("Video ended or failed to read frame.")
                self.tracking_active = False  # Reset tracking flag if video ends or fails
        else:
            if self.processor.cap:
                self.processor.cap.release()
            self.processor.cap = None
            self.tracking_active = False  # Ensure tracking flag is reset when there's no more video

    def display_image(self, frame):
        """Displays the processed frame on the canvas."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(320, 240, image=image, anchor=tk.CENTER)
        self.canvas.image = image


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTrackerApp(root, "Person Tracking with YOLO")
    root.mainloop()
