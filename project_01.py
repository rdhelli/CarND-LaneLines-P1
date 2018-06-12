# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
# %matplotlib inline
import time
import math

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, sy, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    avg_right_theta = 0
    avg_right_b = 0
    avg_right_m = 0
    avg_left_theta = 0
    avg_left_b = 0
    avg_left_m = 0
    num_right = 0
    num_left = 0
    try:
        for line in lines:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)  # x1,y1,x2,y2
            theta = math.atan2(y2-y1,x2-x1) #
            b = y1 - math.tan(theta)*x1
            # b2 = y2 - math.tan(theta)*x2
            if theta > 0.4:
                avg_right_theta += theta
                avg_right_b += b
                num_right += 1
            elif theta < -0.4:
                avg_left_theta += theta
                avg_left_b += b
                num_left += 1
    except TypeError:
        pass
    if num_left > 0:
        avg_left_theta /= num_left
        avg_left_m = math.tan(avg_left_theta)
        avg_left_b /= num_left
        #left_lower
        y_ll = int(sy)
        x_ll = int((sy-avg_left_b)/avg_left_m)
        # left_upper
        y_lu = int(sy/2*1.19)
        x_lu = int((sy/2*1.19-avg_left_b)/avg_left_m)
        # drawing lines
        cv2.line(img, (x_ll,y_ll),(x_lu,y_lu), [0,255,255], 4)
    if num_right > 0:
        avg_right_theta /= num_right
        avg_right_m = math.tan(avg_right_theta)
        avg_right_b /= num_right
        # right_lower
        y_rl = int(sy)
        x_rl = int((sy-avg_right_b)/avg_right_m)
        # right_upper
        y_ru = int(sy/2*1.19)
        x_ru = int((sy/2*1.19-avg_right_b)/avg_right_m)
        #drawing lines
        cv2.line(img, (x_rl,y_rl),(x_ru,y_ru), [0,255,255], 4)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, sy):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # print("num of lines: ",len(lines))
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, sy)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1, γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

# printing out some stats and plotting
# print('This image is:', type(image), 'with dimensions:', image.shape)
# if you wanted to show a single color channel image called 'gray', for example
# call as plt.imshow(gray, cmap='gray')
# plt.imshow(image)

# list images available for testing
# print(os.listdir("test_images/"))

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    gray = grayscale(image)
    blurred = gaussian_blur(gray,5)
    edges = canny(blurred,75,175)
    #edges = canny(blurred,75,175)
    sx, sy = image.shape[1],image.shape[0]
    vertices = np.array([[(.06*sx,sy),(sx/2*.95, sy/2*1.19), (sx/2*1.06, sy/2*1.19), (.97*sx,sy)]], dtype=np.int32)
    #vertices = np.array([[(.16*sx,.9*sy),(sx/2*.95, sy/2*1.19), (sx/2*1.06, sy/2*1.19), (.86*sx,.9*sy)]], dtype=np.int32)
    masked_edges = region_of_interest(edges,vertices)
    lines = hough_lines(masked_edges,1,np.pi/180,15,70,50, sy)
    #lines = hough_lines(masked_edges,2,np.pi/180,15,70,50, sy)
    lanelines = weighted_img(lines,image)
    return lanelines

def debug_process_image():
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    for im_name in os.listdir("test_images/"):
        # reading in an image
        print("processing ", im_name, "...")
        image = mpimg.imread('test_images/{}'.format(im_name))
        ### COPY of process_image(image) function
        gray = grayscale(image)
        blurred = gaussian_blur(gray,5)
        edges = canny(blurred,75,175)
        #edges = canny(blurred,75,175)
        sx, sy = image.shape[1],image.shape[0]
        vertices = np.array([[(.06*sx,sy),(sx/2*.95, sy/2*1.19), (sx/2*1.06, sy/2*1.19), (.97*sx,sy)]], dtype=np.int32)
        #vertices = np.array([[(.16*sx,.9*sy),(sx/2*.95, sy/2*1.19), (sx/2*1.06, sy/2*1.19), (.86*sx,.9*sy)]], dtype=np.int32)
        masked_edges = region_of_interest(edges,vertices)
        lines = hough_lines(masked_edges,1,np.pi/180,15,70,50, sy)
        #lines = hough_lines(masked_edges,2,np.pi/180,15,70,50, sy)
        lanelines = weighted_img(lines,image)
        ###
        for i,coords in enumerate(vertices[0]):
            cv2.line(lanelines,(vertices[0][i][0],vertices[0][i][1]),(vertices[0][(i+1)%len(vertices[0])][0],vertices[0][(i+1)%len(vertices[0])][1]),(0,0,255),2)
        mpimg.imsave("test_images_output/{}_01_gray.png".format(im_name), gray, cmap='gray')
        mpimg.imsave("test_images_output/{}_02_blurred.png".format(im_name), blurred, cmap='gray')
        mpimg.imsave("test_images_output/{}_03_edges.png".format(im_name), edges, cmap='gray')
        mpimg.imsave("test_images_output/{}_04_masked_edges.png".format(im_name), masked_edges, cmap='gray')
        mpimg.imsave("test_images_output/{}_05_lines.png".format(im_name), lines)
        mpimg.imsave("test_images_output/{}_06_lanelines.png".format(im_name), lanelines)
    print("done processing")

debug_process_image()
    
def process_video():
    for vid_name in os.listdir("test_videos/"):
        white_output = 'test_videos_output/{}'.format(vid_name)
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
        clip1 = VideoFileClip("test_videos/{}".format(vid_name))#.subclip(0,5)
        white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
        del clip1

# process_video()

#def process_image(image):
#    # NOTE: The output you return should be a color image (3 channel) for processing video below
#    # TODO: put your pipeline here,
#    # you should return the final output (image where lines are drawn on lanes)
#    gray = grayscale(image)
#    blurred = gaussian_blur(gray,5)
#    edges = canny(blurred,75,175)
#    #edges = canny(blurred,75,175)
#    sx, sy = image.shape[1],image.shape[0]
#    vertices = np.array([[(.06*sx,sy),(sx/2*.95, sy/2*1.19), (sx/2*1.06, sy/2*1.19), (.97*sx,sy)]], dtype=np.int32)
#    #vertices = np.array([[(.16*sx,.9*sy),(sx/2*.95, sy/2*1.19), (sx/2*1.06, sy/2*1.19), (.86*sx,.9*sy)]], dtype=np.int32)
#    masked_edges = region_of_interest(edges,vertices)
#    lines = hough_lines(masked_edges,1,np.pi/180,15,70,50, sy)
#    #lines = hough_lines(masked_edges,2,np.pi/180,15,70,50, sy)
#    lanelines = weighted_img(lines,image)
#    return lanelines