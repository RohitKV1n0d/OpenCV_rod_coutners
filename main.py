import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture('video.mp4')

roi_x = 317
roi_y = 116
roi_width = 1079
roi_height = 949

previous_frame = None
rod_gaps_count = 0
MOVEMENT_THRESHOLD = 1000
NO_MOTION_FRAMES = 50  # Number of consecutive frames without movement to consider rod has passed

motion_detected = False
no_motion_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract the ROI from the frame
    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_frame is None:
        previous_frame = gray
        continue

    frame_diff = cv2.absdiff(previous_frame, gray)
    thresh_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh_diff = cv2.dilate(thresh_diff, None, iterations=2)
    
    movement_pixels = cv2.countNonZero(thresh_diff)
    
    if movement_pixels > MOVEMENT_THRESHOLD:
        motion_detected = True
        no_motion_counter = 0  # Reset the counter when movement is detected
    elif motion_detected:
        no_motion_counter += 1

    # If we've had several consecutive frames without motion, consider the rod passed and increment the count
    if motion_detected and no_motion_counter >= NO_MOTION_FRAMES:
        rod_gaps_count += 1
        motion_detected = False
        no_motion_counter = 0  # Reset for next rod

    previous_frame = gray

    cv2.putText(frame, f"Count: {rod_gaps_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total count of rods passed: {rod_gaps_count}")
