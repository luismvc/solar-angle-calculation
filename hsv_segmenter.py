import cv2
import numpy as np


def nothing(x):
    pass


def segment_image_with_hsv(image):
    # Convert the image to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a window for trackbars
    cv2.namedWindow("Trackbars")

    # Create trackbars for adjusting the HSV values
    cv2.createTrackbar("H_min", "Trackbars", 0, 179, nothing)  # Hue min
    cv2.createTrackbar("H_max", "Trackbars", 179, 179, nothing)  # Hue max
    cv2.createTrackbar("S_min", "Trackbars", 0, 255, nothing)  # Saturation min
    cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)  # Saturation max
    cv2.createTrackbar("V_min", "Trackbars", 0, 255, nothing)  # Value min
    cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)  # Value max

    while True:
        # Get trackbar positions for HSV values
        h_min = cv2.getTrackbarPos("H_min", "Trackbars")
        h_max = cv2.getTrackbarPos("H_max", "Trackbars")
        s_min = cv2.getTrackbarPos("S_min", "Trackbars")
        s_max = cv2.getTrackbarPos("S_max", "Trackbars")
        v_min = cv2.getTrackbarPos("V_min", "Trackbars")
        v_max = cv2.getTrackbarPos("V_max", "Trackbars")

        # Define the lower and upper bounds for HSV
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # Create a mask by applying the HSV range
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)

        # Display the original image, mask, and result
        cv2.imshow("Original Image", image)
        cv2.imshow("HSV Mask", mask)
        cv2.imshow("Segmented Image", result)

        # Break the loop when the 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


# Example usage
image = cv2.imread("data/testing_img/2022-02-26_11_22.jpg")  # Load the input image
# image = cv2.imread("data/testing_img/2022-02-26_16_22.jpg")  # Load the input image
segment_image_with_hsv(image)
