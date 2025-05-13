import cv2

# Test different camera indices to find DroidCam
for index in range(20):  # Test the first 5 indices
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera found at index {index}")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {index}", frame)
            cv2.waitKey(1000)  # Display for 1 second
        cap.release()
        
# destroy all windows after pressing esc
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()