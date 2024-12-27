from re import DEBUG

import cv2
import numpy as np
from pysolar.solar import *

from src.utils import load_data, log, split_string

DEBUG = True


# Image class
class Image:
    def __init__(self, image_path=None, id=None):
        """
        Initializes the Image class.

        :param image_path: Optional path to the image file to load.
        """

        self.image = None
        self.id = id
        self.ch = 1
        self.w = 0
        self.h = 0
        self.cx = 0
        self.cy = 0

        self.offset_cx = 50  # 10
        self.offset_cy = 40  # -10

        if image_path:
            self.load_image(image_path)

    def load_image(self, image_path):
        """
        Loads an image from the given file path.

        :param image_path: Path to the image file.
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.h, self.w, self.ch = self.image.shape

        self.cx = (self.w // 2) + self.offset_cx
        self.cy = (self.h // 2) + self.offset_cy

        if DEBUG:
            log.warn(f"Image loaded from {image_path}")
            log.warn(f"shape: {self.h, self.w, self.ch}")

    def set_img(self, img, id=None):
        self.image = np.copy(img)

        self.h, self.w, self.ch = img.shape

        if id:
            self.id = id

    def is_image_loaded(self):
        """
        Checks if an image is loaded.

        :return: True if an image is loaded, False otherwise.
        """
        return self.image is not None

    def get_image(self):
        """
        Returns the current image.

        :return: The image.
        """
        return self.image

    def get_image_shape(self):
        """
        Returns the shape of the current image.

        :return: Shape of the image (height, width, channels).
        """
        if self.image is not None:
            return self.image.shape
        else:
            print("No image to get shape!")
            return None

    def display_image(self, window_name="Image"):
        """
        Displays the current image in a window.

        :param window_name: The name of the window.
        """
        if self.image is not None:
            cv2.imshow(window_name, self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No image to display!")


# Image processing Class
class img_proc:
    def __init__(self, config_param):

        self.parameters = config_param
        self.data_path = config_param["filePath"]

        self.n_imgs = 0
        self.imgs = []
        self.dome_imgs = []
        self.corrected_imgs = []
        self.sun_mask_imgs = []
        self.sun_imgs = []

        self.radius_dome = 0

    def load_data(self):
        # Loading data
        fileNames = load_data(self.data_path)
        log.info("\tImages names loaded")

        self.n_imgs = len(fileNames)

        for img_file in fileNames:
            # Initialize the Image object with an image
            id = split_string(img_file)
            img = Image(self.data_path + "/" + img_file, id=id)

            self.imgs.append(img)

    def display_one_img(self, ind):
        self.imgs[ind].display_image("Test_Img")

    def display_img_dome(self, ind):
        self.dome_imgs[ind].display_image("Dome test Img")

    def get_img(self, ind):

        return self.imgs[ind].get_image()

    # Some auxiliar functions
    # def draw_circle(self, image, cx, cy, radius_=1, thick=2):
    def draw_circle(self, ind, radius_=1, thick=2):

        if DEBUG:
            log.warn("Drawing the dome circle")

        image = self.get_img(ind)
        cx = self.imgs[ind].cx
        cy = self.imgs[ind].cy

        # Center coordinates
        center_coordinates = (cx, cy)

        # Radius of circle
        radius = radius_

        # Blue color in BGR
        color = (255, 0, 255)

        # Line thickness of 2 px
        thickness = thick

        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of "thick" px
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
        image = cv2.circle(image, center_coordinates, 2, color, thickness)

    def crop_img(self, radius=1):

        self.radius_dome = radius

        for data in self.imgs:
            # dome image
            dome_img_ = np.zeros(
                (2 * self.radius_dome, 2 * self.radius_dome, 3), dtype=np.uint8
            )
            dome_img_ = data.image[
                data.cy - self.radius_dome : data.cy + self.radius_dome,
                data.cx - self.radius_dome : data.cx + self.radius_dome,
            ]

            mask1 = np.zeros_like(dome_img_)
            mask1 = cv2.circle(
                mask1,
                (self.radius_dome, self.radius_dome),
                self.radius_dome,
                (255, 255, 255),
                -1,
            )
            dome_img_ = cv2.bitwise_and(dome_img_, mask1)

            dome_img = Image()
            dome_img.set_img(dome_img_, data.id)

            self.dome_imgs.append(dome_img)

        # self.display_img_dome(ind=0)

    def spherize_image_using_undistort(self) -> int:
        """
        This function applies a "spheronization" effect to an image using an undistortion technique.
        Using appropriate camera calibration parameters, we simulate a spherical effect that can
        "warp" the image as if it were wrapped around a sphere.
        """

        r = 1524  # contanste de aberracion esferica
        rr = 710 - 100
        Cc0, Cc1, Cc2, Cc3 = 8.0, 0.0, 414, 572

        K = np.array([[r, 0.0, rr + 00.0], [0.0, r, rr + 0.0], [0.0, 0.0, 1.0]])
        D = np.array([[Cc0], [Cc1], [Cc2], [Cc3]])

        # undistorted_img = None
        for dome_img in self.dome_imgs:

            # cv2.imshow("dome_img", dome_img.image)
            # cv2.waitKey(0)

            dim = (dome_img.w, dome_img.h)

            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, dim, np.eye(3), balance=0.0
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, dim, cv2.CV_16SC2
            )
            undistorted_img = cv2.remap(
                dome_img.image,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            corrected_img = Image()
            corrected_img.set_img(undistorted_img, dome_img.id)

            self.corrected_imgs.append(corrected_img)

        # cv2.imshow("corrected", undistorted_img)

        return 0

    # def segment_sun(
    #     self,
    #     min_radius=20,
    #     max_radius=50,
    #     h_min=0,
    #     h_max=0,
    #     s_min=0,
    #     s_max=255,
    #     v_min=255,
    #     v_max=255,
    # ):
    #     """
    #     Segments the Sun from an image within the sky-dome using color thresholding and circular Hough Transform.

    #     Parameters:
    #     - min_radius (int): Minimum radius of the Sun in pixels.
    #     - max_radius (int): Maximum radius of the Sun in pixels.
    #     - h_min (int): Minimum hue value for Sun color detection in HSV space.
    #     - h_max (int): Maximum hue value for Sun color detection in HSV space.
    #     - s_min (int): Minimum saturation value for Sun color detection in HSV space.
    #     - v_min (int): Minimum value (brightness) for Sun color detection in HSV space.

    #     """

    #     # Convert the image to HSV (Hue, Saturation, Value) color space
    #     for count, imgs_data in enumerate(self.corrected_imgs):
    #         image = imgs_data.image
    #         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #         # Define the color range for detecting the Sun (typically yellowish or white)
    #         lower_bound = np.array([h_min, s_min, v_min])
    #         upper_bound = np.array([h_max, s_max, v_max])

    #         # Apply the color threshold to create a binary mask for the Sun
    #         mask = cv2.inRange(hsv, lower_bound, upper_bound)

    #         # Use a bitwise AND to isolate the Sun from the original image
    #         result = cv2.bitwise_and(image, image, mask=mask)

    #         kernel = np.ones((5, 5), np.uint8)
    #         opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    #         cv2.imshow("sun_seg_res" + str(count), opening)

    #         # Convert the result to grayscale for further processing
    #         gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    #         # Apply Gaussian blur to reduce noise and improve circle detection
    #         blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    #         # Apply the Hough Circle Transform to detect circles (the Sun)
    #         circles = cv2.HoughCircles(
    #             blurred,
    #             cv2.HOUGH_GRADIENT,
    #             dp=1.2,
    #             minDist=20,
    #             param1=50,
    #             param2=30,
    #             minRadius=min_radius,
    #             maxRadius=max_radius,
    #         )

    #         sun_center = None
    #         sun_radius = None
    #         sun_mask = None

    #         if circles is not None:
    #             # Convert the (x, y) coordinates and radius of the circles to integers
    #             circles = np.uint16(np.around(circles))
    #             for circle in circles[0, :]:
    #                 center = (circle[0], circle[1])  # Center of the circle
    #                 radius = circle[2]  # Radius of the circle

    #                 # Create a mask for the Sun
    #                 sun_mask = np.zeros_like(mask)
    #                 cv2.circle(sun_mask, center, radius, (255), thickness=-1)

    #                 # Return the center and radius of the Sun
    #                 sun_center = center
    #                 sun_radius = radius

    #                 # Draw the circle on the image for visualization (optional)
    #                 cv2.circle(image, center, radius, (0, 255, 0), 4)
    #                 cv2.rectangle(
    #                     image,
    #                     (center[0] - 5, center[1] - 5),
    #                     (center[0] + 5, center[1] + 5),
    #                     (0, 128, 255),
    #                     3,
    #                 )
    #         else:
    #             log.warn("No circles detected")

    #         cv2.imshow("sun_seg" + str(count), image)
    def segment_sun(
        self,
        min_sun_area=500,
        max_sun_area=10000,
        min_ellipsoid_ratio=0.5,
        max_ellipsoid_ratio=2.0,
    ):
        """
        Segments the Sun from an image, considering an ellipsoidal form, using connected components and morphological operations.

        Parameters:
        - image (numpy.ndarray): Input image in which the Sun needs to be segmented.
        - min_sun_area (int): Minimum area (in pixels) for a connected component to be considered as the Sun.
        - max_sun_area (int): Maximum area (in pixels) for a connected component to be considered as the Sun.
        - min_ellipsoid_ratio (float): Minimum aspect ratio for the detected ellipsoid.
        - max_ellipsoid_ratio (float): Maximum aspect ratio for the detected ellipsoid.

        Returns:
        - sun_mask (numpy.ndarray): Binary mask where the Sun is segmented.
        - sun_center (tuple): (x, y) coordinates of the Sun's center.
        - sun_axes (tuple): (major_axis, minor_axis) lengths of the detected Sun ellipsoid.
        - result_image (numpy.ndarray): The input image with the segmented Sun highlighted.
        """

        for count, imgs_data in enumerate(self.corrected_imgs):
            image = imgs_data.image
            # Convert the image to HSV (Hue, Saturation, Value) color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define the color range for detecting the Sun (typically yellowish or white)
            h_min, h_max = 0, 50  # Hue range for yellow (adjust if needed)
            s_min, s_max = 0, 255  # Saturation range for bright yellow
            v_min, v_max = 255, 255  # Value range for bright areas (Sun)

            # Apply the color threshold to create a binary mask for the Sun
            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

            # Apply morphological operations to clean up small artifacts
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel
            )  # Closing operation to remove small holes
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel
            )  # Opening operation to remove small objects

            # Find connected components in the binary mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            # cv2.imshow("sun_seg_mask" + str(count), mask)
            # cv2.waitKey()

            sun_mask_img = Image()
            sun_mask_img.set_img(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), imgs_data.id)
            self.sun_mask_imgs.append(sun_mask_img)

            sun_mask = np.zeros_like(mask)
            sun_center = None
            sun_axes = None
            result_image = image.copy()

            for i in range(1, num_labels):  # Ignore the background component (label 0)
                # Get the stats for the component
                x, y, w, h, area = stats[i]

                # Filter components based on area to avoid small reflections or noise
                if min_sun_area <= area <= max_sun_area:
                    # Create a mask for the Sun's component
                    sun_mask[labels == i] = 255

                    # Get the contour of the connected component
                    contour = np.array(
                        cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )[-2]
                    )

                    # Fit an ellipse to the contour (which might be ellipsoidal)
                    if len(contour) > 0:
                        ellipse = cv2.fitEllipse(contour[0])
                        center, axes, angle = ellipse

                        # Calculate the aspect ratio (major axis / minor axis)
                        major_axis, minor_axis = axes
                        aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 0

                        # Check if the aspect ratio is within the expected range for the Sun (typically near 1 for circular)
                        if min_ellipsoid_ratio <= aspect_ratio <= max_ellipsoid_ratio:
                            sun_center = center
                            sun_axes = (major_axis, minor_axis)

                            # Draw the ellipse on the result image
                            cv2.ellipse(result_image, ellipse, (255, 0, 0), 3)

                            break  # Assuming the largest component is the Sun, stop after the first detection
                        else:
                            log.warn(f"No sun detected: {count}")

            # cv2.imshow("sun_seg" + str(count), result_image)

            sun_img = Image()
            sun_img.set_img(result_image, imgs_data.id)
            self.sun_imgs.append(sun_img)

        return 0

    def create_mosaic(self, resize_dim=(300, 300)) -> None:
        """
        Create a mosaic from a set of images.

        """

        grid_shape = (self.n_imgs, 5)

        images = []
        for i in range(grid_shape[0]):
            img_resized = cv2.resize(self.imgs[i].image, resize_dim)
            images.append(img_resized)

            img_resized = cv2.resize(self.dome_imgs[i].image, resize_dim)
            images.append(img_resized)

            img_resized = cv2.resize(self.corrected_imgs[i].image, resize_dim)
            images.append(img_resized)

            img_resized = cv2.resize(self.sun_mask_imgs[i].image, resize_dim)
            images.append(img_resized)

            img_resized = cv2.resize(self.sun_imgs[i].image, resize_dim)
            images.append(img_resized)

        mosaic_rows = []

        for i in range(grid_shape[0]):
            row_images = images[i * grid_shape[1] : (i + 1) * grid_shape[1]]
            mosaic_rows.append(np.hstack(row_images))

        mosaic_image = np.vstack(mosaic_rows)

        cv2.imshow("mosaic", mosaic_image)


def process(parameters: dict) -> int:
    """Main processing function"""

    log.info("Processing images...")

    img_proc_ = img_proc(parameters)
    img_proc_.load_data()
    # img_proc_.display_one_img(ind=0)

    # visualizing the best circle that fits the dome
    # img_proc_.draw_circle(ind=0, radius_=parameters["radius_dome"])

    img_proc_.crop_img(radius=parameters["radius_dome"])

    img_proc_.spherize_image_using_undistort()

    img_proc_.segment_sun()

    img_proc_.create_mosaic()

    cv2.waitKey(0)

    return 0
