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


def process(parameters: dict) -> int:
    """Main processing function"""

    log.info("Processing images...")

    img_proc_ = img_proc(parameters)
    img_proc_.load_data()
    img_proc_.display_one_img(ind=0)

    # visualizing the best circle that fits the dome
    # img_proc_.draw_circle(ind=0, radius_=parameters["radius_dome"])

    img_proc_.crop_img(radius=parameters["radius_dome"])

    img_proc_.spherize_image_using_undistort()

    cv2.waitKey(0)

    return 0
