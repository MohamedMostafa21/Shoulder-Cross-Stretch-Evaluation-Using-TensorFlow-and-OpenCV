import tkinter as tk
from tkinter import ttk
import subprocess

def start_detection():
    subprocess.Popen(["python", "real_time_predict.py"], shell=True)

def exit_program():
    # Close the GUI window
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Shoulder Cross Stretch Evaluation")
root.geometry("500x500")  # Set the window size

# Set the window background color
root.configure(bg="#3498db")

# Create a canvas
canvas = tk.Canvas(root, width=500, height=500, bg="#3498db", highlightthickness=0)
canvas.pack()

# Draw background
canvas.create_rectangle(50, 50, 450, 450, fill="#ecf0f1", outline="")

# Start button
start_button = tk.Button(root, text="Start Evaluation", command=start_detection, bg="#2ecc71", fg="white", font=("Helvetica", 14, "bold"))
start_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

# Exit button
exit_button = tk.Button(root, text="Exit", command=exit_program, bg="#e74c3c", fg="white", font=("Helvetica", 14, "bold"))
exit_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

# Run the Tkinter event loop
root.mainloop()
