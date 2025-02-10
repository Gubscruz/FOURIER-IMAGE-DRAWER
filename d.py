import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import json

# Global variables for drawing
drawing = False           # True when the left mouse button is pressed
drawn_points = []         # List to store drawn (x,y) points in order
bg_image = None           # The background image
temp_image = None         # A working copy for drawing

def mouse_callback(event, x, y, flags, param):
    global drawing, drawn_points, temp_image
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        drawn_points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            drawn_points.append((x, y))
            if len(drawn_points) >= 2:
                cv2.line(temp_image, drawn_points[-2], drawn_points[-1], (0, 0, 255), thickness=2)
            cv2.imshow("Draw", temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        drawn_points.append((x, y))
        if len(drawn_points) >= 2:
            cv2.line(temp_image, drawn_points[-2], drawn_points[-1], (0, 0, 255), thickness=2)
        cv2.imshow("Draw", temp_image)

def select_background_image():
    root = tk.Tk()
    root.withdraw()
    # Pass filetypes as a tuple of patterns to avoid macOS errors.
    file_path = filedialog.askopenfilename(
        title="Select Background Image", 
        filetypes=[("Image Files", ("*.png", "*.jpg", "*.jpeg", "*.bmp")), ("All Files", "*")]
    )
    return file_path

def main():
    global bg_image, temp_image, drawn_points
    img_path = select_background_image()
    if not img_path:
        print("No image selected.")
        exit(0)
    bg_image = cv2.imread(img_path)
    if bg_image is None:
        print("Failed to load image.")
        exit(1)
    temp_image = bg_image.copy()
    
    cv2.namedWindow("Draw")
    cv2.setMouseCallback("Draw", mouse_callback)
    cv2.imshow("Draw", temp_image)
    print("Draw your outline on the image (you may draw in multiple strokes).")
    print("Press 'Enter' when finished, or 'q' to quit.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)
    cv2.destroyWindow("Draw")
    print(f"Total drawn points: {len(drawn_points)}")
    
    # If the drawing is not closed, close it by connecting the end to the start.
    if np.linalg.norm(np.array(drawn_points[0]) - np.array(drawn_points[-1])) > 10:
        drawn_points.append(drawn_points[0])
    
    # Save the drawn points to a JSON file.
    with open("drawn_points.json", "w") as f:
        json.dump(drawn_points, f)
    print("Saved drawn points to 'drawn_points.json'.")

if __name__ == "__main__":
    main()