import cv2
import numpy as np

class RODCounter:
    def __init__(self, file_path, show_roi=False, show_video=True):
        if not show_video:
            print("Couting Rods")
        self.file_path = file_path
        self.show_roi = show_roi
        self.show_video = show_video
        self.cap = cv2.VideoCapture(self.file_path)
        self.previous_frame = None
        self.rod_gaps_count = 0
        self.MOVEMENT_THRESHOLD = 1000
        self.NO_MOTION_FRAMES = 50
        self.motion_detected = False
        self.no_motion_counter = 0

    def extract_roi(self, frame, roi_x, roi_y, roi_width, roi_height):
        if self.show_roi:
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
        return frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    def process_frames(self):
        roi_x, roi_y = 317, 116
        roi_width, roi_height = 1079, 949

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            roi = self.extract_roi(frame, roi_x, roi_y, roi_width, roi_height)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if self.previous_frame is None:
                self.previous_frame = gray
                continue

            frame_diff = cv2.absdiff(self.previous_frame, gray)
            thresh_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh_diff = cv2.dilate(thresh_diff, None, iterations=2)
            movement_pixels = cv2.countNonZero(thresh_diff)

            if movement_pixels > self.MOVEMENT_THRESHOLD:
                self.motion_detected = True
                self.no_motion_counter = 0
            elif self.motion_detected:
                self.no_motion_counter += 1

            if self.motion_detected and self.no_motion_counter >= self.NO_MOTION_FRAMES:
                self.rod_gaps_count += 1
                self.motion_detected = False
                self.no_motion_counter = 0

            self.previous_frame = gray

            if self.show_video:
                cv2.putText(frame, f"Count: {self.rod_gaps_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        if self.show_video:
            cv2.destroyAllWindows()
        
        return f"Total count of rods passed: {self.rod_gaps_count}"

counter = RODCounter('video.mp4', show_roi=True, show_video=False)
print(counter.process_frames())
