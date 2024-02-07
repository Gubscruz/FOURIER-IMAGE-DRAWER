# noise reduction using gaussian filter
import cv2
import numpy as np


def process_image(image_path):
    img = cv2.imread(image_path)
    
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply gaussian filter
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # edge detection
    edges = cv2.Canny(blurred, 100, 200)
    
    cv2.imwrite('edges.png', edges)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


process_image('aviao_full.jpeg')
