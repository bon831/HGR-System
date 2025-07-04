from picamera2 import Picamera2
import cv2
import os
import time

# Define classes and their respective folders
classes = [
    "all_fingers_up", "all_fingers_down", "thumb_up", "thumb_index_up",
    "thumb_index_middle_up", "thumb_index_middle_ring_up", "thumb_pinky_up",
    "thumb_down", "thumb_index_down", "thumb_index_middle_down",
    "thumb_index_middle_ring_down", "peace_sign"
]

# Base directory for saving dataset
base_dir = "crop_dataset"
os.makedirs(base_dir, exist_ok=True)

# Create folders for each class
for class_name in classes:
    class_path = os.path.join(base_dir, class_name)
    os.makedirs(class_path, exist_ok=True)

# Key mappings for the classes
key_mappings = {
    ord('0'): 0, ord('1'): 1, ord('2'): 2, ord('3'): 3, ord('4'): 4,
    ord('5'): 5, ord('6'): 6, ord('7'): 7, ord('8'): 8, ord('9'): 9,
    ord('a'): 10, ord('b'): 11, ord('c'): 12
}

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (540, 580), "format": "RGB888"})
picam2.configure(config)
picam2.start()

print("Camera started with default resolution.")
print("Press the key corresponding to the class number to save an image:")
for i, class_name in enumerate(classes):
    key_label = str(i) if i < 10 else chr(ord('a') + (i - 10))
    print(f"Press '{key_label}' for '{class_name}'")
print("Press 'q' to quit.")

while True:
    # Capture a frame
    frame = picam2.capture_array()

    # Display the frame
    cv2.imshow("Capture Gesture", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Check if 'q' is pressed to quit
    if key == ord('q'):
        print("Exiting.")
        break

    # Check if the key corresponds to a valid class
    if key in key_mappings:
        class_index = key_mappings[key]
        class_name = classes[class_index]

        # Save the captured image
        timestamp = int(time.time())    # use unix timestamp to avoid overwriting the existing file (in seconds)
        filename = f"{base_dir}/{class_name}/{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")

# Stop the camera and close windows
picam2.stop()
cv2.destroyAllWindows()
