from manim import *
import numpy as np
import json

def resample_contour(points, num_points):
    pts = np.array(points, dtype=np.float64)
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
    """
    Compute the discrete Fourier transform (DFT) of the uniformly resampled contour.
    Represent the contour as complex numbers (x + i*y) after centering.
    Reorder the coefficients in symmetric order: 0, 1, -1, 2, -2, ...
    Returns a list of tuples: (frequency, coefficient).
    """
    N = len(points)
    z = points[:, 0] + 1j * points[:, 1]
    z = z - np.mean(z)
    dft = np.fft.fft(z) / N
    freqs = np.fft.fftfreq(N, d=1.0/N)
    # Build symmetric ordering: 0, 1, -1, 2, -2, ...
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
    """
    Given time t (0 <= t <= 1) and a list of Fourier coefficients (in symmetric order),
    compute the endpoint of the epicycle chain and record each epicycle's segment.
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

class FourierEpicyclesScene(Scene):
    def construct(self):
        # Load drawn points from the JSON file.
        with open("drawn_points.json", "r") as f:
            points = json.load(f)
        points = np.array(points, dtype=np.float64)
        
        # Resample the drawn contour uniformly (adjust number of points as needed).
        resampled = resample_contour(points, 1000)
        
        # Compute the Fourier coefficients in symmetric order.
        coeffs = compute_dft(resampled)
        
        # Create a ValueTracker for time t (from 0 to 1).
        t_tracker = ValueTracker(0)
        
        def get_epicycles():
            x, y = 0, 0
            epicycles_group = VGroup()
            for freq, coef in coeffs:
                prev_x, prev_y = x, y
                angle = 2 * np.pi * freq * t_tracker.get_value() + np.angle(coef)
                radius = abs(coef)
                x += radius * np.cos(angle)
                y += radius * np.sin(angle)
                circle = Circle(radius=radius, color=BLUE, stroke_width=1).move_to(np.array([prev_x, prev_y, 0]))
                line = Line(np.array([prev_x, prev_y, 0]), np.array([x, y, 0]), color=YELLOW)
                epicycles_group.add(circle, line)
            return epicycles_group, np.array([x, y, 0])
        
        # Create a traced path of the endpoint.
        traced_path = TracedPath(lambda: get_epicycles()[1], stroke_color=RED, stroke_width=2)
        epicycles_mobs = always_redraw(lambda: get_epicycles()[0])
        
        self.add(epicycles_mobs, traced_path)
        self.play(t_tracker.animate.set_value(1), run_time=10, rate_func=linear)
        self.wait()