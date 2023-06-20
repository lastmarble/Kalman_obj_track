import numpy as np
import cv2


def detect(frame, debugMode):
    img = frame
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper range of the red color in HSV color space
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])
    red_mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine the two red masks
    red_mask = red_mask1 + red_mask2

    # Apply a morphological opening to remove noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the red mask
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    # Draw the contours on the original image
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            centers.append(np.array([[x], [y]]))
            cv2.imshow('contours', img)
    return centers
    