# Solar Angle Calculation from Images

This project provides Python-based tools to process a set of images, segment the apparent Sun's position, and calculate solar angles, such as azimuth and elevation, based on the detected Sun position. This project is particularly useful for applications such as solar energy systems, astronomical research, and solar radiation studies.

## Overview

The goal of this project is to detect the Sun's apparent position in images and compute solar angles, which are crucial for understanding the Sun's trajectory in the sky. The angles calculated include:

- **Solar Azimuth**: The compass direction of the Sun (measured clockwise from North).
- **Solar Elevation**: The angle of the Sun above the horizon.

These angles can help in applications like solar panel positioning, weather prediction, and space observation.

## Features

- **Image Segmentation**: Detects the Sun's apparent position in each image.
- **Solar Angle Calculation**: Computes azimuth and elevation based on the Sun's detected position and known geographic parameters (e.g., location and time).
- **Flexible Workflow**: Can process any set of images and can be adapted for different geographic locations and times.
- **Output Visualization**: Includes the ability to overlay the Sun�s position on the images and visualize solar angles.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Image Preprocessing](#image-preprocessing)
   - [Running the Image Processing Script](#running-the-image-processing-script)
   - [Visualizing Results](#visualizing-results)
3. [Algorithm Details](#algorithm-details)
4. [Requirements](#requirements)
5. [Example Output](#example-output)
6. [Notes](#notes)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

---

## Installation

To run this project, you'll need Python 3.13 or higher. Follow the steps below to set up the environment and install dependencies:

### 1. Clone the repository

```bash

git clone https://github.com/luismvc/solar-angle-calculation.git
cd solar-angle-calculation
```
