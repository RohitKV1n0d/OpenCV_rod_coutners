import cv2

# Callback function to handle mouse events
def select_roi(event, x, y, flags, param):
    global drawing, top_left_pt, bottom_right_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        cv2.rectangle(img, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        cv2.imshow('Select ROI', img)
        print(f"ROI Coordinates: {top_left_pt} {bottom_right_pt}")


# Read the video and get the first frame
cap = cv2.VideoCapture('video.mp4')
ret, img = cap.read()

# Check if frame read successfully
if not ret:
    print("Error reading video")
    exit()

drawing = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)

# Show the frame and bind the mouse callback function
cv2.namedWindow('Select ROI')
cv2.setMouseCallback('Select ROI', select_roi)

# Display the frame and wait for a key press
cv2.imshow('Select ROI', img)
cv2.waitKey(0)

# If an ROI is selected, display it on the frame
if top_left_pt != (-1, -1) and bottom_right_pt != (-1, -1):
    roi_img = img[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
    cv2.imshow('Selected ROI', roi_img)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
