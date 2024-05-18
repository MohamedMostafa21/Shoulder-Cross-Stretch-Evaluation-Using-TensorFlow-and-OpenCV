import cv2
import os
import time

# Directories to save the images
correct_dir = 'data/train/correct'
incorrect_dir = 'data/train/incorrect'

# Create directories if they do not exist
os.makedirs(correct_dir, exist_ok=True)
os.makedirs(incorrect_dir, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the number of images to capture and the count as well
num_images = 70
count_correct = 0
count_incorrect = 0

capturing_correct = False
capturing_incorrect = False

print("Press 'c' to start capturing 'correct' images every 3 seconds.")
print("Press 'i' to start capturing 'incorrect' images every 3 seconds.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break
    
    cv2.imshow('Capture Images', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c'):
        capturing_correct = True
        capturing_incorrect = False
        start_time = time.time()
    
    elif key == ord('i'):
        capturing_correct = False
        capturing_incorrect = True
        start_time = time.time()
    
    elif key == ord('q'):
        break

    current_time = time.time()
    
    if capturing_correct and count_correct < num_images and (current_time - start_time) >= 2:
        img_name = os.path.join(correct_dir, f'correct3_{count_correct}.jpg')
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        count_correct += 1
        start_time = current_time
    
    if capturing_incorrect and count_incorrect < num_images and (current_time - start_time) >= 2:
        img_name = os.path.join(incorrect_dir, f'incorrect3_{count_incorrect}.jpg')
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        count_incorrect += 1
        start_time = current_time

    if count_correct >= num_images and count_incorrect >= num_images:
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

print(f"Captured {count_correct} 'correct' images and {count_incorrect} 'incorrect' images.")
