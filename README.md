# Sick plants detector

<div align="center">

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/opencv-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Overview 

This project is meant to be a fast way to spot and count plants affected by Fusarium wilt in large greenhouses.
A drone with a camera working on visible-light spectrum was used to shoot the images, 
which are courtesy of [Applied Drone Innovations B.V.](https://applieddroneinnovations.nl/) 
The program is based upon the OpenCV library and uses various filters and image processing techniques to spot yellow-reddish leaves.

![Sick Plants Detector screenshot](resources/readme/screenshot.png)

## Run the code

1) Open a shell window and clone the project.
    ```bash
    $ git clone https://github.com/mstrocchi/plantalyzer.git
    ```

2) Get into the project's root.
    ```bash
    $ cd plantalyzer
    ``` 

3) Install the required packages.
    ```bash
    $ pip install -r requirements.txt 
    ``` 

4) Run it with Python 3.
    ```bash
    $ python3 sick-plants-detector.py
    ```

## Author

**Mattia Strocchi** - [m.strocchi@student.tudelft.nl](mailto:m.strocchi@student.tudelft.nl) 

## License

This project is available under the MIT license. See the [LICENSE](https://github.com/mstrocchi/plantalyzer/blob/master/LICENSE.md) file for more info.

