import cv2
import numpy as np
import json

MASK_RCNN_WEIGHTS = "mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
MASK_RCNN_CONFIG = "mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
CONFIDENCE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.3

def get_person_contour(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found.")
    H, W = image.shape[:2]
    net = cv2.dnn.readNetFromTensorflow(MASK_RCNN_WEIGHTS, MASK_RCNN_CONFIG)
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    person_contour = None
    best_area = 0

    # Iterate through detections to find a person mask
    for i in range(boxes.shape[2]):
        score = boxes[0, 0, i, 2]
        class_id = int(boxes[0, 0, i, 1])

        print("Detection score:", score , " class_id:", class_id)
        if score > CONFIDENCE_THRESHOLD and class_id == 1:

                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                boxW = endX - startX
                boxH = endY - startY
                mask = masks[i, 0]
                mask = cv2.resize(mask, (boxW, boxH))
                mask = (mask > MASK_THRESHOLD).astype(np.uint8) * 255

                debug_mask =  np.zeros((H, W), dtype=np.uint8)
                debug_mask[startY:endY, startX:startX+boxW] = mask

                cv2.imshow("Mask", debug_mask)
                cv2.waitKey(0)

                full_mask = np.zeros((H, W), dtype=np.uint8)
                full_mask[startY:endY, startX:endX] = mask

                # Morphological closing to clean up the mask
                kernel = np.ones((5, 5), np.uint8)
                full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    if area > best_area:
                        best_area = area
                        person_contour = largest_contour

    # Fallback: Canny edge detection if no Mask R-CNN contour
    if person_contour is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            person_contour = max(contours, key=cv2.contourArea)
        else:
            raise ValueError("No contour found.")

    # Convert from [N,1,2] to [N,2]
    person_contour = person_contour[:, 0, :]
    return person_contour

def resample_contour(contour, num_points):
    pts = contour.astype(np.float32)
    distances = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(distances)))
    total_length = cumulative[-1]
    evenly_spaced = np.linspace(0, total_length, num_points)
    resampled = []
    for d in evenly_spaced:
        idx = np.searchsorted(cumulative, d)
        if idx >= len(pts):
            idx = len(pts) - 1
        if cumulative[idx] == d or idx == 0:
            resampled.append(pts[idx])
        else:
            t = (d - cumulative[idx - 1]) / (cumulative[idx] - cumulative[idx - 1])
            point = pts[idx - 1] * (1 - t) + pts[idx] * t
            resampled.append(point)
    return np.array(resampled)

def compute_dft(points):
    N = len(points)
    z = points[:, 0] + 1j * points[:, 1]
    dft = np.fft.fft(z) / N
    freqs = np.fft.fftfreq(N, d=1)
    coeffs = []
    for k in range(N):
        coeffs.append({
            "re": float(dft[k].real),
            "im": float(dft[k].imag),
            "freq": int(np.round(freqs[k])),
            "amp": float(np.abs(dft[k])),
            "phase": float(np.angle(dft[k]))
        })
    coeffs.sort(key=lambda c: c["amp"], reverse=True)
    return coeffs

if __name__ == "__main__":
    image_path = "./person.jpg"
    contour = get_person_contour(image_path)
    resampled = resample_contour(contour, 500)
    coeffs = compute_dft(resampled)
    with open("coeffs.json", "w") as f:
        json.dump(coeffs, f)

    # Create a blank canvas of the same size as the original image
    # We'll read the image again just to get the dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to re-open the image.")
    H, W = image.shape[:2]

    # Black background
    blank_canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Draw the extracted contour on the blank canvas
    contour_to_draw = contour.reshape((-1, 1, 2)).astype(np.int32)
    cv2.drawContours(blank_canvas, [contour_to_draw], -1, (0, 255, 0), 2)

    # Show only the contour against the black background
    cv2.imshow("Contour Only", blank_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()