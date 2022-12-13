## Image Operation Visualizer
Tool to visualize various mathematical and arithmetic operations on image matrices. Created for Cristian Rodriguez's Fall 2022 MATH235 Linear Algebra class.
![App](https://github.com/dgorbunov/matrix_math_visualizer/blob/main/app.png)

## Installation:
1. Install Python from the Python website if you don't already have it.
2. Install the following dependencies using these commands:
  - Run `pip install numpy`
  - Run `pip install opencv-python`
  - Run `pip install PyQt5`

## Instructions for Use:
1. Run `python3 main.py` to run the program.
2. A display window with a GUI should pop up. If you run into an error here, make sure you update Python and all the required libraries to their latest version.
3. If you wish to modify the images used, replace `image.png` and `image2.png` in `/images`, and rename the new images with those names. Input images should be square and the same dimensions; they will automatically be resized to fit these criteria if they do not meet them.
4. The output image is written to `images/output.png` on every operation.

## Thanks
Default images created by OpenAI's DALL-E 2 🧸.
