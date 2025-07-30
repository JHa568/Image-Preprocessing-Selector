import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

images_loc = "./images/"

entries = {}

original = cv2.imread(images_loc + "image100.jpg", cv2.IMREAD_GRAYSCALE)
original = cv2.resize(original, (300, 300))  # Resize for better visibility
if original is None:
    raise FileNotFoundError("Make sure 'your_image.jpg' exists in the current directory.")

# Create main window
root = tk.Tk()
root.geometry("800x600")
root.title("Image Processing Adjuster")

# Output image label
label = tk.Label(root, width=300, height=300)
label.pack()

# --- Function to update image ---
def update_image(*args):
    # Read current slider values
    block_size = block_slider.get()
    if block_size % 2 == 0:
        block_size += 1  # must be odd
    if block_size < 3:
        block_size = 3

    C = c_slider.get()
    dilate_ksize = dilate_slider.get()
    erode_ksize = erode_slider.get()
    open_ksize = open_slider.get()
    close_ksize = close_slider.get()
    grad_ksize = gradient_slider.get()
    tophat_ksize = tophat_slider.get()
    blackhat_ksize = blackhat_slider.get()

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        original, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )

    # Apply dilation
    if dilate_ksize > 1:
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Apply erosion
    if erode_ksize > 1:
        kernel = np.ones((erode_ksize, erode_ksize), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
    
     # Opening
    if open_ksize > 1:
        kernel = np.ones((open_ksize, open_ksize), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Closing
    if close_ksize > 1:
        kernel = np.ones((close_ksize, close_ksize), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Morphological Gradient
    if grad_ksize > 1:
        kernel = np.ones((grad_ksize, grad_ksize), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

    # Top Hat (original - opening)
    if tophat_ksize > 1:
        kernel = np.ones((tophat_ksize, tophat_ksize), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)

    # Black Hat (closing - original)
    if blackhat_ksize > 1:
        kernel = np.ones((blackhat_ksize, blackhat_ksize), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_BLACKHAT, kernel)

    # Convert to ImageTk and update display
    img = ImageTk.PhotoImage(Image.fromarray(thresh))
    label.imgtk = img
    label.config(image=img)

def save_parameters():
    params = {
        "block_size": block_slider.get(),
        "C": c_slider.get(),
        "dilate_ksize": dilate_slider.get(),
        "erode_ksize": erode_slider.get(),
        "open_ksize": open_slider.get(),
        "close_ksize": close_slider.get(),
        "grad_ksize": gradient_slider.get(),
        "tophat_ksize": tophat_slider.get(),
        "blackhat_ksize": blackhat_slider.get()
    }
    with open("parameters.txt", "w") as f:
        for key, value in params.items():
            f.write(f"{value} - {key}\n")
    print("Parameters saved to parameters.txt")

# --- Sliders ---
control_frame = tk.Frame(root)
control_frame.pack()

block_slider = tk.Scale(control_frame, from_=3, to=99, label="Block Size (odd)", orient="horizontal", command=update_image)
block_slider.set(11)

c_slider = tk.Scale(control_frame, from_=-20, to=20, label="C (Bias)", orient="horizontal", command=update_image)
c_slider.set(2)

dilate_slider = tk.Scale(control_frame, from_=1, to=15, label="Dilation Kernel", orient="horizontal", command=update_image)
dilate_slider.set(1)

erode_slider = tk.Scale(control_frame, from_=1, to=15, label="Erosion Kernel", orient="horizontal", command=update_image)
erode_slider.set(1)

open_slider = tk.Scale(control_frame, from_=1, to=15, label="Opening Kernel", orient="horizontal", command=update_image)
open_slider.set(1)

close_slider = tk.Scale(control_frame, from_=1, to=15, label="Closing Kernel", orient="horizontal", command=update_image)
close_slider.set(1)

gradient_slider = tk.Scale(control_frame, from_=1, to=15, label="Gradient Kernel", orient="horizontal", command=update_image)
gradient_slider.set(1)

tophat_slider = tk.Scale(control_frame, from_=1, to=15, label="Top Hat Kernel", orient="horizontal", command=update_image)
tophat_slider.set(1)

blackhat_slider = tk.Scale(control_frame, from_=1, to=15, label="Black Hat Kernel", orient="horizontal", command=update_image)
blackhat_slider.set(1)

button = tk.Button(root, text="Save", command=save_parameters)
blackhat_slider.set(1)
button.pack()

# Grid sliders 4 per row
sliders = [
    block_slider, c_slider, dilate_slider, erode_slider,
    open_slider, close_slider, gradient_slider, tophat_slider, blackhat_slider
]

for idx, slider in enumerate(sliders):
    row = idx // 4
    col = idx % 4
    slider.grid(row=row, column=col, sticky="ew", padx=5, pady=5)

# Make columns expand equally
for col in range(4):
    control_frame.grid_columnconfigure(col, weight=1)

# Initial update
update_image()

root.mainloop()