import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def combine_contours(contours):
    """
    Given a list of contours (each as a NumPy array of shape (n,1,2)),
    convert them to lists of (x, y) and join them into one continuous outline.
    The algorithm starts with the longest contour and then repeatedly
    attaches the remaining contour whose starting or ending point is closest
    to the current endpoint.
    """
    # Convert each contour to a list of (x, y) points.
    contour_lists = [c.reshape(-1, 2).tolist() for c in contours]
    if not contour_lists:
        return []

    # Choose the contour with the maximum arc length as starting contour.
    lengths = [cv2.arcLength(np.array(c, dtype=np.float32), True) for c in contour_lists]
    start_idx = np.argmax(lengths)
    combined = contour_lists[start_idx]
    used = {start_idx}
    current_endpoint = combined[-1]

    while len(used) < len(contour_lists):
        best_idx = None
        best_dist = float('inf')
        best_reverse = False
        # Look through all remaining contours.
        for i, pts in enumerate(contour_lists):
            if i in used:
                continue
            # Distance from current endpoint to the start of this contour.
            dist_start = np.linalg.norm(np.array(current_endpoint) - np.array(pts[0]))
            # Distance from current endpoint to the end of this contour.
            dist_end = np.linalg.norm(np.array(current_endpoint) - np.array(pts[-1]))
            if dist_start < best_dist:
                best_dist = dist_start
                best_idx = i
                best_reverse = False
            if dist_end < best_dist:
                best_dist = dist_end
                best_idx = i
                best_reverse = True
        if best_idx is None:
            break
        best_contour = contour_lists[best_idx]
        if best_reverse:
            best_contour = best_contour[::-1]
        # Optionally, you could insert an interpolated connector here.
        combined.extend(best_contour)
        current_endpoint = best_contour[-1]
        used.add(best_idx)
    return combined

def process_image(image_path):
    """
    Process the image:
      - Read the image and apply 10 bilateral filters.
      - Convert to grayscale and detect edges with Canny.
      - Find external contours and combine them into one continuous outline.
      - Display the combined outline for verification.
    Returns the combined outline as a NumPy array of shape (N, 2) (floats).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image: {image_path}")

    # Apply bilateral filtering 10 times
    for _ in range(10):
        image = cv2.bilateralFilter(image, 3, 20, 50)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 130, apertureSize=3)

    # Use RETR_EXTERNAL so that only the outer boundaries are found.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found!")
    
    # Combine all external contours into one continuous outline.
    combined_points = combine_contours(contours)
    combined_points = np.array(combined_points, dtype=np.float32)
    
    # Optionally, show the combined outline.
    blank = np.zeros_like(gray)
    cv2.polylines(blank, [combined_points.astype(np.int32)], isClosed=False, color=255, thickness=1)
    cv2.imshow("Combined Contour", blank)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return combined_points

def resample_contour(points, num_points):
    """
    Resample the ordered contour points so they are uniformly spaced.
    'points' is a NumPy array of shape (N, 2).
    """
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(dists)))
    total_length = cumulative[-1]
    sample_dists = np.linspace(0, total_length, num_points)
    resampled = []
    for d in sample_dists:
        idx = np.searchsorted(cumulative, d)
        if idx == 0:
            resampled.append(points[0])
        elif idx >= len(points):
            resampled.append(points[-1])
        else:
            t = (d - cumulative[idx-1]) / (cumulative[idx] - cumulative[idx-1])
            point = points[idx-1]*(1-t) + points[idx]*t
            resampled.append(point)
    return np.array(resampled)

def compute_dft(points):
    """
    Compute the discrete Fourier transform (DFT) of the resampled contour.
    The contour is represented as complex numbers (x + i*y), then centered.
    Returns a list of Fourier coefficients (frequency, coefficient).
    """
    N = len(points)
    z = points[:, 0] + 1j * points[:, 1]
    # Center the contour
    z = z - np.mean(z)
    dft = np.fft.fft(z) / N
    freqs = np.fft.fftfreq(N, d=1.0/N)
    coeffs = [(freqs[k], dft[k]) for k in range(N)]
    # Optionally, sort by amplitude if desired.
    coeffs.sort(key=lambda c: abs(c[1]), reverse=True)
    return coeffs

def epicycles(t, coeffs):
    """
    Given time t (0 to 1) and the list of Fourier coefficients,
    compute the endpoint and each epicycle segment.
    Returns:
      - endpoint: (x, y)
      - positions: list of tuples ((x0, y0), (x1, y1), amplitude)
    """
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

def animate_epicycles(coeffs, num_frames=200):
    """
    Animate the Fourier epicycles drawing the outline.
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    total_amp = sum(abs(coef) for _, coef in coeffs)
    ax.set_xlim(-total_amp*1.1, total_amp*1.1)
    ax.set_ylim(-total_amp*1.1, total_amp*1.1)
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
    # Process the image and extract the entire outline (all external contours combined)
    combined_points = process_image("./img/draw.jpg")
    print("Number of combined outline points:", len(combined_points))
    
    # Resample the outline to a fixed number of points (e.g., 500)
    resampled = resample_contour(combined_points, 1000)
    
    # Compute the Fourier coefficients from the resampled outline
    coeffs = compute_dft(resampled)
    print("Number of Fourier coefficients:", len(coeffs))
    
    # Animate the drawing via Fourier epicycles
    animate_epicycles(coeffs)