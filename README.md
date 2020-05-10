# Sick plants detector

<div align="center">

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/opencv-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Overview 

This project is meant to be a fast way to spot and count plants affected by Fusarium wilt in large greenhouses using drones with a camera working on visible-light spectrum. 
The program is based upon OpenCV library and uses various filters and image processing techniques to spot yellow-reddish leaves.

![Plantalyzer screenshot](resources/readme/screenshot.png)

Aerial photos are courtesy of [Applied Drone Innovations B.V.](https://applieddroneinnovations.nl/)

## Run the code

1) Open a shell window and clone the project.
    ```bash
    $ git clone https://github.com/mstrocchi/sick-plants-detector.git
    ```

2) Get into the project's root.
    ```bash
    $ cd sick-plants-detector
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

**FID2WAV** is available under the MIT license. See the [LICENSE](https://github.com/mstrocchi/fid-to-wav/blob/master/LICENSE.md) file for more info.

