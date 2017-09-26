## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[cal]: ./output_images/cal.png "cal"
[undist]: ./output_images/undist.png "undist"
[bin]: ./output_images/bin.png "bin"
[bin_seg]: ./output_images/bin_seg.png "bin_seg"
[pers]: ./output_images/pers.png "pers"
[fit]: ./output_images/fit.png "fit"
[marked]: ./output_images/lane_marked.png "marked"
[pipeline]: ./output_images/pipeline.png "pipeline"
[pipeline2]: ./output_images/pipeline2.png "pipeline2"
[pipeline3]: ./output_images/pipeline3.png "pipeline3"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I implemented checkerboard images to get the camera matrix in [cal.py] (detector / cal.py).

* Obtain the corner point of the checkerboard images and save it as coordinates in the 2d image plane. : `` `DistortionCorrector._get_img_points ()` `
* 2d Obtain the world world coordinate corresponding to the coordinates in the image plane. : `` `DistortionCorrector._get_obj_points ()` `
* 2d Save the coordinates in the image plane and the coordinates in 3d world coordinate as class member variables.
* When an image requiring distortion correction is input, use the coordinates obtained above to obtain the camera matrix and correct the distortion for the input image. : `` `DistortionCorrec

![alt text][cal]

The above figure is the result of distortion correction test. Run [cal.py] (detector / cal.py) to see the results.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of s-channel intensity thresholing and edge image.

![alt text][bin]

* Obtain binary image by intensity and edge map using canny edge detector separately, as shown in the middle of the figure above.
* It detects lane pixels only when there is an edge pixel in the left and right direction for the active pixel in binay image. Through this process, it is possible to minimize the misunderstanding of the shadow as lane pixel as shown below.
	* The process of detecting lane pixels using binary image and edge map is implemented in [framework.py] (detector / lane / framework.py).

![alt text][bin_seg]



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I have manually set the source and destination points as in the code below.

```python
src_points = np.array([(250, 700), (1075, 700), (600, 450), (685, 450)]).astype(np.float32)

w, h = dst_size
x_offset = 300
y_offset = 50
dst_points = np.array([(x_offset, h-y_offset),
                       (w-x_offset, h-y_offset),
                       (x_offset, y_offset),
                       (w-x_offset, y_offset)]).astype(np.float32)

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 250, 700      | 300, 710      | 
| 1075, 700     | 980, 710      |
| 600, 450      | 300, 50       |
| 685, 450      | 980, 50       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][pers]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I selected the lane pixel used for the polynomial fitting in the sliding window and performed the second order polynomial fitting.

![alt text][fit]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented the process of finding the radius of curvature in the curv.py `` Curvature class``.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The process of converting a fitted lane curve into a perspective of the original image and displaying the lane area is implemented in the `` LaneMarker class` of [pers.py] (detector / curve / pers.py).


![alt text][marked]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I implemented an algorithm to detect the lane area in this project. The most time-consuming research I have done is to detect lane pixels in the original perspective view. Especially, in the image with mixed shadows, it was difficult to extract only the lane pixels by gradient and intensity thresholding.

![alt text][pipeline]

So I decided to combine both binary images and edge maps separately, as in the above pipeline.

![alt text][pipeline3]
(blue : bright pixel, red : edge pixel)

As shown in the figure above, only pixels with edges in both directions on the horizon line of bright pixels are detected as lane pixels. This will minimize shadowing mistakes.

![alt text][pipeline2]

However, even with this method, recognition results were not good in trees with many images or images with shadows. Here are my ideas for improvement.

* How to use time series information
	* This method reflects the recognition result from the previous image to the prediction in the current frame.
* Using prior knowledge that the left and right lanes are parallel when fitting the lane curve
* How to use machine learning algorithm
	* However, this method requires a dataset labeled for lane pixels.

