import json
import numpy as np
from manim import *

class FourierEpicyclesScene(Scene):
    def construct(self):
        with open("coeffs.json", "r") as f:
            coeffs = json.load(f)
        max_time = 2 * PI
        time_tracker = ValueTracker(0)
        def get_epicycles():
            x = 0
            y = 0
            epicycles = VGroup()
            for term in coeffs:
                prev_x = x
                prev_y = y
                freq = term["freq"]
                radius = term["amp"] * 200
                angle = freq * time_tracker.get_value() + term["phase"]
                x += radius * np.cos(angle)
                y += radius * np.sin(angle)
                circle = Circle(radius=radius, color=BLUE, stroke_width=1).move_to(np.array([prev_x, prev_y, 0]))
                line = Line(np.array([prev_x, prev_y, 0]), np.array([x, y, 0]), color=YELLOW)
                epicycles.add(circle, line)
            return epicycles, np.array([x, y, 0])
        path = TracedPath(lambda: get_epicycles()[1], stroke_color=RED, stroke_width=2)
        epicycles_mob = always_redraw(lambda: get_epicycles()[0])
        self.add(epicycles_mob, path)
        self.play(time_tracker.animate.set_value(max_time), run_time=20, rate_func=linear)
        self.wait()

if __name__ == "__main__":
    from manim import config
    config.media_width = "75%"
    scene = FourierEpicyclesScene()
    scene.render()