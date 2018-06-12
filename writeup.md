# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve.jpg_01_gray.png "Grayscale"
[image2]: ./test_images_output/solidWhiteCurve.jpg_02_blurred.png "Blurred"
[image3]: ./test_images_output/solidWhiteCurve.jpg_03_edges.png "Edges"
[image4]: ./test_images_output/solidWhiteCurve.jpg_04_masked_edges.png "Region of interest"
[image5]: ./test_images_output/solidWhiteCurve.jpg_05_lines.png "Lines"
[image6]: ./test_images_output/solidWhiteCurve.jpg_06_lanelines.png "Lane line segments"
[image7]: ./test_images_output/solidWhiteCurve.jpg_lanelines.png "Lane lines (continuous)"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline utilized the method of the classes with some fine-tuning and tricks. First the image is converted to grayscale, then a gaussian blurring follows, with a Kernel size of 5×5 pixels, to reduce noise related sudden increments in the gradients.
![alt text][image1]
![alt text][image2]
The edges are then recoverable with the canny algorithm. I increased the thresholds to reduce unrelated content, which worked acceptably on the yellow lane lines too.
![alt text][image3]
Then I created a polygon to mask only the region of interest. I used relative dimensions so that the method stays robust over different image sizes. The slight assymetry suggests that the camera was not positioned right in the middle.
![alt text][image4]
Lines can be recovered with the help of a Hough transformation. Once again, I increased the thresholds to filter out smaller segments, as the dashed lane line can now be connected with greater line gaps (as long as classification is not required). This method helped limiting the number of line segments and stabilizing the detection pipeline.
![alt text][image5]
To be able to build 2 continuous lane approximation lines, I modified the draw_lines() function. I classified the segments based on their theta angles into 3 categories: part of the right lane line if significantly positive, part of the left lane line if significantly negative, and a disturbance if close to zero. This method works practically over regular driving circumstances. The left and right segments are then averaged separately based on their slope and offset values. The resulting line equations were used to mark the endpoints in the region of interest.
![alt text][image7]

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
