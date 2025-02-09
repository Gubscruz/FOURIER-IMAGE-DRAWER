import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import filedialog

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
    file_path = filedialog.askopenfilename(title="Select Background Image")
    return file_path

def resample_contour(points, num_points):
    pts = np.array(points, dtype=np.float32)
    dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(dists)))
    total_length = cumulative[-1]
    sample_dists = np.linspace(0, total_length, num_points)
    resampled = []
    for d in sample_dists:
        idx = np.searchsorted(cumulative, d)
        if idx == 0:
            resampled.append(pts[0])
        elif idx >= len(pts):
            resampled.append(pts[-1])
        else:
            t = (d - cumulative[idx-1]) / (cumulative[idx] - cumulative[idx-1])
            point = pts[idx-1]*(1-t) + pts[idx]*t
            resampled.append(point)
    return np.array(resampled)

def compute_dft(points):
    N = len(points)
    z = points[:, 0] + 1j * points[:, 1]
    z = z - np.mean(z)
    dft = np.fft.fft(z) / N
    freqs = np.fft.fftfreq(N, d=1.0/N)
    # Reorder coefficients in symmetric order: 0, 1, -1, 2, -2, ...
    indices = [0]
    for i in range(1, (N+1)//2):
        indices.append(i)
        indices.append(-i)
    if N % 2 == 0:
        indices.append(N//2)
    reordered = []
    for k in indices:
        idx = k if k >= 0 else N + k
        reordered.append((freqs[idx], dft[idx]))
    return reordered

def epicycles(t, coeffs):
    x, y = 0, 0
    positions = []
    for freq, coef in coeffs:
        prev_x, prev_y = x, y
        angle = 2 * np.pi * freq * t + np.angle(coef)
        dx = abs(coef) * np.cos(angle)
        dy = abs(coef) * np.sin(angle)
        x += dx
        y += dy
        positions.append(((prev_x, prev_y), (x, y), abs(coef)))
    return (x, y), positions

def animate_epicycles(coeffs, num_frames=300):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    total_amp = sum(abs(coef) for _, coef in coeffs)
    ax.set_xlim(-total_amp * 1.1, total_amp * 1.1)
    ax.set_ylim(-total_amp * 1.1, total_amp * 1.1)
    ax.axis('off')
    
    line_objs = []
    circle_objs = []
    for _ in coeffs:
        line_obj, = ax.plot([], [], 'r-', lw=1)
        line_objs.append(line_obj)
        circle_obj, = ax.plot([], [], 'g--', lw=0.5)
        circle_objs.append(circle_obj)
    trace_line, = ax.plot([], [], 'b-', lw=2)
    trace = []
    
    def init():
        for line_obj in line_objs:
            line_obj.set_data([], [])
        for circle_obj in circle_objs:
            circle_obj.set_data([], [])
        trace_line.set_data([], [])
        return line_objs + circle_objs + [trace_line]
    
    def update(frame):
        t = frame / num_frames
        endpoint, positions = epicycles(t, coeffs)
        for i, ((x0, y0), (x1, y1), amp) in enumerate(positions):
            line_objs[i].set_data([x0, x1], [y0, y1])
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = x0 + amp * np.cos(theta)
            circle_y = y0 + amp * np.sin(theta)
            circle_objs[i].set_data(circle_x, circle_y)
        trace.append(endpoint)
        trace_np = np.array(trace)
        trace_line.set_data(trace_np[:, 0], trace_np[:, 1])
        return line_objs + circle_objs + [trace_line]
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init,
                                  blit=True, interval=20, repeat=False)
    ax.invert_yaxis()
    plt.show()

if __name__ == "__main__":
    # Select a background image using a file dialog.
    img_path = select_background_image()
    if not img_path:
        print("No image selected.")
        exit(0)
    bg_image = cv2.imread(img_path)
    if bg_image is None:
        print("Failed to load image.")
        exit(1)
    
    # Optionally, resize the background image if it is too large.
    temp_image = bg_image.copy()
    
    # Create a window for drawing.
    cv2.namedWindow("Draw")
    cv2.setMouseCallback("Draw", mouse_callback)
    cv2.imshow("Draw", temp_image)
    print("Draw your outline on the image (you may draw in multiple strokes).")
    print("Press 'Enter' when finished, or 'q' to quit.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key to finish drawing.
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)
    cv2.destroyWindow("Draw")
    print(f"Total drawn points: {len(drawn_points)}")
    
    # If the drawing is not closed, close it by connecting the end to the start.
    if np.linalg.norm(np.array(drawn_points[0]) - np.array(drawn_points[-1])) > 10:
        drawn_points.append(drawn_points[0])
    
    # Resample the drawn points uniformly.
    resampled = resample_contour(drawn_points, 1000)
    
    # Compute the Fourier coefficients from the resampled (and centered) points.
    coeffs = compute_dft(resampled)
    
    # Animate the Fourier epicycles reconstruction.
    animate_epicycles(coeffs)