# Shoulder-Cross-Stretch-Evaluation-Using-TensorFlow-and-OpenCV
<br> This project aims to develop a computer vision application using convolutional neural networks (CNN) to evaluate the correctness of shoulder cross stretch exercises. The application utilizes TensorFlow for model training and OpenCV for real-time image processing.
<br>
# Requirements
* Check the requirements.txt 
# Running the Model
to run the model using your webcam, just type
```bash
python app.py
```
<br>
Then a simple GUI window will open.<br>
Press "Start Evaluation" to use the model, or "Exit" to leave.<br>
To exit the ongoing detection window, press 'q'<br>

# Directory
<pre>
│  README.md
│  app.py
│  evaluate_model.py
│  gathering_data.py
│  real_time_predict.py
│  requirements.txt
│  testing.py
│  train_model.py
│  30_Demo.mp4
└─models
   │  shoulder_stretch_model_optimized.keras
   │  
   │
   └─train_mod
</pre>
### app.py
The GUI app that allows you to start detection or exit the model.
### train_model.py
The main model training script for Shoulder Cross Stretch Exercise.
  

