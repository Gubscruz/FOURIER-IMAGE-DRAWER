import cv2
import numpy as np

def goFilterCV(image_path):
    # Read the image from the given path
    draft = cv2.imread(image_path)
    if draft is None:
        raise IOError(f"Could not open or find the image: {image_path}")
    
    # Dimensions (width = ow, height = oh in the Java code)
    oh, ow = draft.shape[:2]
    
    # 1) Apply multiple bilateral filters
    #    (diameter=3, sigmaColor=20, sigmaSpace=50), repeated 10 times
    for _ in range(10):
        draft = cv2.bilateralFilter(draft, 3, 20, 50)
    
    # 2) Convert to grayscale
    gray = cv2.cvtColor(draft, cv2.COLOR_BGR2GRAY)
    
    # 3) Edge detection with Canny
    #    (threshold1=100, threshold2=130, apertureSize=3)
    edges = cv2.Canny(gray, 100, 130, apertureSize=3)
    
    # 4) Contour detection
    #    Using RETR_LIST (find all contours) and CHAIN_APPROX_NONE (store all points)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print("Number of contours found:", len(contours))
    
    # 5) Polynomial approximation of each contour
    #    - We replicate the Java code’s approach using approxPolyDP with epsilon=3.
    #    - The Java code increments 'iterazione' for each contour and 'cc' for each point.
    #      We’ll store everything in a list `E`, where each entry is (x, y, iteration).
    E = []            # Equivalent to int[][] E in Java code
    iteration = 0     # 'iterazione' from Java code
    cc = -1           # 'cc' from Java code (tracks cumulative points)
    
    for contour in contours:
        iteration += 1
        # Approximate the contour to a polygon
        # (3 is the epsilon used in the Java code’s cvApproxPoly)
        approx = cv2.approxPolyDP(contour, 3, closed=False)
        
        # Each 'approx' point is shape (1,2), so we extract (x, y)
        for pt in approx:
            cc += 1
            x, y = pt[0]
            # Store x, y, and the iteration index
            E.append((x, y, iteration))
    
    # At this point:
    #  - 'draft' holds the final filtered image
    #  - 'edges' holds the Canny edges
    #  - 'contours' is the list of raw contours
    #  - 'E' is the list of approximated points (like E[][] in the Java code)
    
    # If you want to see the edges result in a window (like the returned PImage in Java),
    # you can uncomment these lines:
    cv2.imshow("Bilateral + Canny result", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Return whatever is relevant:
    # - The edges image (like returning the final PImage in Java)
    # - The list of approximated points
    return edges, E

# Example usage:
if __name__ == "__main__":
    edges_image, points_list = goFilterCV("elevador.jpg")
    print("Total approximated points:", len(points_list))
    # Each entry in points_list is (x, y, iteration_index).