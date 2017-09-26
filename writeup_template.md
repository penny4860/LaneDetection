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

[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

저는 checkerboard images 를 사용해서 camera matrix를 구하는 과정을 [cal.py](detector/cal.py)에 구현하였습니다. 

* checkerboard images의 corner point를 구해서 이를 2d image plane 에서의 좌표로 저장한다. : ```DistortionCorrector._get_img_points()```
* 2d image plane 에서의 좌표에 대응하는 3d world coordinate 좌표를 구한다. : ```DistortionCorrector._get_obj_points()```
* 2d image plane 에서의 좌표들과 3d world coordinate에서의 좌표들을 class member 변수로 저장한다.
* distortion correction 이 필요한 image가 입력되었을 때 위에서 구한 좌표들을 이용해서 camera matrix를 구하고 input image 에 대해서 distortion을 보정한다. : ```DistortionCorrector.run()```

![alt text][cal]

위 그림은 distortion correction을 test한 결과입니다. [cal.py](detector/cal.py)를 실행하면 결과를 볼 수 있습니다.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

저는 s-channel intensity의 thresholing 과 edge image를 조합해서 사용했습니다.

![alt text][bin]

* 위 figure의 가운데 그림처럼 intensity 에 의한 binary image 와 canny edge detector를 이용한 edge map을 따로 구합니다.
* binay image 에서의 active pixel 에 대해서 왼쪽과 오른쪽 방향의 비슷한 거리에 edge pixel이 있는 경우만 lane pixel로 검출합니다. 이러한 과정을 통해 아래의 그림과 같이 그림자를 lane pixel로 오인식하는 경우를 최소화 할 수 있습니다.
	* binary image 와 edge map을 이용해서 lane pixel을 검출하는 과정은 [framework.py](detector/lane/framework.py) 에 구현되어있습니다.

![alt text][bin_seg]



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

저는 아래의 code와 같이 source points와 destination points를 manual로 정하였습니다.

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

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
