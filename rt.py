"""
This code was largely adapted from the upcoming Udacity Self-Driving Nanodegree Project 0.

Adaptations are moving this toward a real-time video processing algorithm that would allow
for real-time video to be processed and displayed on a HUD in a vehicle.

Additionally, common functions have been generalized and modularized
to help the open-source community.
"""

# IMPROT LIBS
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import sys


"""------------ FUNCTION DEFINITIONS ------------"""

def backspace(n):
    # print((b'\x08' * n).decode(), end='') # use \x08 char to go back
    print('\r' * n, end='')

def grayscale(img):
    # Grayscale Transform OpenCV
    # Important to note that resulting image is now only 1 channel
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    # Canny Transform from OpenCV
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_noise(img, kernel_size):
    # Blur, kernal size determines blurriness
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    # Image masking
    # Creating blank mask of zeroes
    mask = np.zeros_like(img)

    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # Grabbing the number of channels
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Returning the image only where mask pixels are non-zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=(225, 0, 255), thickness=10):
    #Draws `lines` with `color` and `thickness`.
    #Lines are drawn on the image inplace (mutates the image).
    global line_no
    if lines is not None: # Ensures that if a given frame has no lines it will still compile
        line_no = len(lines)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    else:
        line_no = 0;

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # `img` should be the output of a Canny transform.
    # Returns an image with hough lines drawn.

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # COLOR.GRAY2BGR puts this back in 3 channel so that colorful lines can be drawn
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return gaussian_noise(line_img, 5) # Applying a blur gives a little more softness to the lines

# Cool Symbols supported in Python 3, avoid otherwise!
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def image_smoothing(processed_image, image_array):
    blank_image = np.zeros((processed_image["h"],processed_image["w"],3),np.float) #Creating blank image size of frame
    for im in image_array: #loop over images in array
        imarr = np.array(im,dtype=np.float)
        blank_image = blank_image + imarr / len(image_array) # take average of all frames
    image_smoothed =  np.array(np.round(blank_image),dtype=np.uint8) # float to int
    return image_smoothed




"""------------- MAIN PROCESSING FLOW ---------------"""

def process_image(img):
    # Image intialization, storing a copy as init_img
    init_img = img
    height = img.shape[0] # Getting image height and width
    width = img.shape[1]

    # Edge Detection
    image = grayscale(img)
    image = gaussian_noise(image, 5)
    image = canny(image, 50, 150)

    #Region of Interest Masking
    region = np.array([[(0,height),(width/2, 4*height/7), (width/2, 4*height/7), (width,height)]], dtype=np.int32)

    image = region_of_interest(image, region)

    #Drawing Hough Lines
    rho = 1 # Distance resolution in pixels of the Hough grid
    theta = np.pi/180 # Angular resolution in radians of the Hough grid
    threshold = 15    # Minimum number of votes (intersections in Hough grid cell)
    min_line_len = 210 # Minimum number of pixels making up a line
    max_line_gap = 160    # Maximum gap in pixels between connectable line segments

    image_lines = hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap)
    # image = weighted_img(image_lines, init_img, α=0.8, β=1., λ=0.)

    return {"image": image, "lines": image_lines, "init": init_img, "h": height, "w": width}





"""--------------- VIDEO PROCESSING ---------------"""

cap = cv2.VideoCapture('solidYellowLeft.mp4')

frame_no = 0
line_no = 0
frame_list = []
time_list = []
line_list = []
image_array = [] # queue of the last five images for smooothing algo

while(cap.isOpened()):
    #print("lines:", str(line_no),end='')
    #backspace(len(str(line_no)))

    ret, frame = cap.read()

    start_time = time.time() # Starting timer performance analysis

    # Catpuring live frames
    if frame is not None:
        processed = process_image(frame)
        image_array.append(processed["lines"]) #adding frames to image buffer
        frame_list.append(frame_no)
        line_list.append(line_no)
        # limiting number of images to be averaged at X
        if len(image_array) > 3:
            del image_array[0]

        # Calling Smoothing Algorithm
        init_img = processed["init"]
        smoothed_lines = image_smoothing(processed,image_array)
        final_image = weighted_img(smoothed_lines, init_img, α=0.8, β=1., λ=0.)

    time_list.append(time.time() - start_time) # Ending timer for performance analysis
    cv2.imshow('frame',final_image)
    frame_no += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





"""-------------- PLOTTING ----------------"""

time_list = np.array(time_list)
frame_list = np.array(frame_list)
line_list = np.array(line_list)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(frame_list, time_list)
ax2 = ax1.twinx()
ax2.plot(frame_list, line_list, 'ro')
plt.show()
