import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

images_loc = "../images/"

entries = {}

original = cv2.imread(images_loc + "sudoku.jpg", cv2.IMREAD_GRAYSCALE) #  
original = cv2.resize(original, (300, 300))  # Resize for better visibility
if original is None:
    raise FileNotFoundError("Make sure 'your_image.jpg' exists in the current directory.")

# Create main window
root = tk.Tk()
root.geometry("800x600")
root.title("Image Processing Adjuster")

# Output image label
pic_frame = tk.Frame(root)
pic_frame.pack(padx=3, pady=5)

label_apply = tk.Label(pic_frame, text="Contour Image", width=300, height=300)
label_apply.pack(side="left", padx=5)

label = tk.Label(pic_frame, text="Filtered Image", width=300, height=300)
label.pack(side="left", padx=5)

is_using_edge_detection = tk.IntVar()
apply_contours = tk.IntVar()

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

    copy_original = original.copy()
    # Apply adaptive threshold
    
    thresh = cv2.cvtColor(copy_original, cv2.COLOR_GRAY2BGR)  # Convert to BGR for contour drawing
    if is_using_edge_detection.get() == 1:
        low_canny_thresh = int(low_canny_thresh_slider.get())
        high_canny_thresh = int(high_canny_thresh_slider.get())
        thresh = cv2.cvtColor(copy_original, cv2.COLOR_BGR2RGB)
        aperture_size = aperture_canny_thresh_slider.get()  # Default aperture size for Canny
        if aperture_size % 2 == 0:
            aperture_size += 1  # must be odd
        print(f"Edge detection: {is_using_edge_detection.get()}, Low: {low_canny_thresh}, High: {high_canny_thresh}, aperture: {aperture_size}")
        thresh = cv2.Canny(thresh, low_canny_thresh, high_canny_thresh, apertureSize=aperture_size)
    else:
        thresh = cv2.adaptiveThreshold(
            copy_original, 255,
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

    if apply_contours.get() == 1:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_for_contours = cv2.cvtColor(copy_original, cv2.COLOR_GRAY2BGR)  # Convert to BGR for contour drawing
        contour_img = cv2.drawContours(image_for_contours, contours, -1, (0, 255, 0), 1)
        contour_img_tk = ImageTk.PhotoImage(Image.fromarray(contour_img))
        label_apply.imgtk = contour_img_tk
        label_apply.config(image=contour_img_tk)
    else:
        no_contour_img = ImageTk.PhotoImage(Image.fromarray(copy_original))
        label_apply.imgtk = no_contour_img
        label_apply.config(image=no_contour_img)

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
        "blackhat_ksize": blackhat_slider.get(),
        "low_canny_thresh": low_canny_thresh_slider.get(),
        "high_canny_thresh": high_canny_thresh_slider.get(),
        "aperture_canny_thresh": aperture_canny_thresh_slider.get()
    }
    with open("parameters.txt", "w") as f:
        for key, value in params.items():
            if "canny" in key and is_using_edge_detection.get() == 1:
                f.write(f"{value} - {key}\n")
            else:
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

check_canny = tk.Checkbutton(control_frame, text="Canny Edge Detection", variable=is_using_edge_detection, command=update_image)

aperture_canny_thresh_slider = tk.Scale(control_frame, from_=3, to=7, label="Aperture", orient="horizontal", command=update_image)
aperture_canny_thresh_slider.set(3)

low_canny_thresh_slider = tk.Scale(control_frame, from_=0, to=255, label="Low Thresh Canny Edge Detect", orient="horizontal", command=update_image)
low_canny_thresh_slider.set(127)

high_canny_thresh_slider = tk.Scale(control_frame, from_=0, to=255, label="High Thresh Canny Edge Detect", orient="horizontal", command=update_image)
high_canny_thresh_slider.set(255)

check_contour = tk.Checkbutton(control_frame, text="apply contours", variable=apply_contours, command=update_image)

button = tk.Button(root, text="Save", command=save_parameters)
blackhat_slider.set(1)
button.pack()

# Grid sliders 4 per row
sliders = [
    block_slider, c_slider, dilate_slider, erode_slider,
    open_slider, close_slider, gradient_slider, tophat_slider, blackhat_slider,
    check_canny, aperture_canny_thresh_slider, low_canny_thresh_slider, high_canny_thresh_slider, check_contour
]

for idx, slider in enumerate(sliders):
    row = idx // 6
    col = idx % 6
    slider.grid(row=row, column=col, sticky="ew", padx=2, pady=5)

# Make columns expand equally
for col in range(2):
    pic_frame.grid_columnconfigure(col, weight=1)

for col in range(6):
    control_frame.grid_columnconfigure(col, weight=1)

# Initial update
update_image()

root.mainloop()