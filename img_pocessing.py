# noise reduction using gaussian filter
import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_image(image_path):
    img = cv2.imread(image_path)
    
    # convert to gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply gaussian filter
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # edge detection
    # edges = cv2.Canny(blurred, 100, 200)


    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    contours
    
    # cv2.imwrite('edges.png', edges)
    # cv2.imshow('Edges', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    for contour in contours:
        x = contour[:, :, 0].reshape(-1,)
        y = -contour[:, :, 1].reshape(-1,)
        ax.plot(x, y)

    plt.show()

#NOTE - try to get the poits of each contour with the closest distance to the next contour

process_image('aviao_vetor.jpeg')
