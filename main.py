import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp


class VideoProcessor:
    def __init__(self):
        # Load YOLO and Pose Estimation
        self.net = cv2.dnn.readNet(r"model/yolov4.weights", r"model/yolov4.cfg")
        self.classes = [line.strip() for line in open(r"model/coco.names", "r")]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cap = None

    def process_frame(self, frame):
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

                box_color = (0, 255, 0)  # Default to green box
                text = action if action != "Unsure" else ""  # Only display action if not 'Unsure'

                if fall_detected:
                    box_color = (0, 0, 255)  # Change to red if fall is detected
                    text = "Falling Down"  # Change text when a fall is detected

                # Ensure the text is displayed within the bounds of the image
                text_location = self.calculate_text_location(x, y, w, h, text)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        return frame

    def detect_fall(self, pose_landmarks):
        # Refine fall detection logic to reduce false positives
        landmarks = pose_landmarks.landmark
        nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
        mid_hip = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y +
                   landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y) / 2
        # A fall is detected if the nose is close to the mid-hip level, suggesting horizontal orientation
        if nose.y > mid_hip - 0.1:
            return True
        return False

    def estimate_action(self, pose_landmarks):
        landmarks = pose_landmarks.landmark

        # Retrieve necessary landmarks
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Calculate knee angles using a defined function
        def calculate_angle(a, b, c):
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

        # Check vertical alignment of hips and ankles
        def is_vertically_aligned(hip, ankle):
            return abs(hip.y - ankle.y) < 0.1  # Tune this threshold based on empirical data

        if (is_vertically_aligned(l_hip, l_ankle) or is_vertically_aligned(r_hip, r_ankle)) and (
                left_knee_angle < 120 or right_knee_angle < 120):
            return 'Sitting'
        elif left_knee_angle > 165 and right_knee_angle > 165:
            return 'Standing'
        elif (left_knee_angle > 140 or right_knee_angle > 140) and (left_knee_angle < 165 or right_knee_angle < 165):
            return 'Walking'
        else:
            return 'Running'

    def calculate_text_location(self, x, y, w, h, text):
        # Calculate text location dynamically based on box position and size
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x + (w - text_size[0]) // 2  # Center text horizontally within the box
        text_y = y + h - 5  # Adjusted to be within the lower part of the box
        return (text_x, text_y)


class VideoTrackerApp:
    def __init__(self, window, window_title):
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

    def open_video(self):
        self.processor.cap = cv2.VideoCapture(filedialog.askopenfilename())
        ret, frame = self.processor.cap.read()
        if ret:
            self.display_image(frame)

    def start_tracking(self):
        if not self.processor.cap.isOpened():
            self.processor.cap = cv2.VideoCapture(filedialog.askopenfilename())
        fps = self.processor.cap.get(cv2.CAP_PROP_FPS)
        self.processor.delay = int(1000 / fps)
        self.update_frame()

    def reset_video(self):
        if self.processor.cap and self.processor.cap.isOpened():
            self.processor.cap.release()
        self.processor.cap = None
        self.canvas.delete("all")
        print("Reset the Video.")

    def update_frame(self):
        if self.processor.cap and self.processor.cap.isOpened():
            ret, frame = self.processor.cap.read()
            if ret:
                self.display_image(self.processor.process_frame(frame))
                self.window.after(self.processor.delay, self.update_frame)
            else:
                self.processor.cap.release()
                self.processor.cap = None
                print("Video ended or failed to read frame.")
        else:
            if self.processor.cap:
                self.processor.cap.release()
            self.processor.cap = None

    def display_image(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(320, 240, image=image, anchor=tk.CENTER)
        self.canvas.image = image


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTrackerApp(root, "Person Tracking with YOLO")
    root.mainloop()
