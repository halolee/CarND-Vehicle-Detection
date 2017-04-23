##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[detection]: ./examples/detection.png
[heatmap]: ./examples/heatmap.png
[HOG]: ./examples/HOG_example.png
[samples]: ./examples/sample.png
[video1]: ./pipeLineOutput.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.
You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the below code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][HOG]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and it turns out the combination of following parameters yields a better for the HOG detection result.

    colorSpace = YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL"

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier training are inside `train_SVC_if_needed` function. In order to improve performance, the function will check whether a previously trained classifier exist in the `svc_picke` file.

In terms of feature selection, I used a combination of color histogram, spatial bin, and HOG in the color space of `YCrCb`. These features are extracted in `extract_features` and then combined.

Normalisation happens in the `data_preparation` and `find_cars)` before the svc prediction.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search has been implemented in `sliding_windows_search` function.

In general, the sliding window will not search in the sky or the bonnet area. And the futher the car, the smaller the scale.

The number is based on experiment result. 
* Scale 1.5 alone can perform quite well on a whole image search
* At scale smaller than 1.0 or larger than 3.0 will generate additional false positive
* More overlapping search could result in false positive

Therefore the end result has been settled down on 2 scales of 1.5 and 2.0. The ystart and ystop position are (400, 540) and (480, 680) respectively.

![alt text][detection]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][samples]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./pipeLineOutput.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. 

To further rejecting out false positive, I store the last detected frame and add it to the new frame's heat map.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][heatmap]
---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are a few ways to improve the project:
* fine tune the features to reduce false positive
* add more scale in the sliding window to narrow down bounding box
* increase cached vehicle detection to smooth detection

On top of that, the processing time for the vechicle is far from real time. It tooks more than 5 minutes for processing a 51 seconds video at 30 frames per seconds.

Performance could be improved by restricing the sliding window search area to only the right 2/3 of the window, but it will lose the generalisation of vechicle in a non-high way situation.

Future work could be using deep learning (i.e. [YOLO](https://pjreddie.com/darknet/yolo/)) to provide a real time object recognition.