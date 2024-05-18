import cv2
import numpy as np
import tensorflow as tf

# Load the trained model with the same architecture used during training
model = tf.keras.models.load_model('models/shoulder_stretch_model_optimized.keras')

# OpenCV video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    img = cv2.resize(frame, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    label = 'Correct' if prediction < 0.5 else 'Incorrect'

    # Display results
    if label == 'Correct':
        cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Shoulder Stretch Evaluation', frame)
    print(prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
