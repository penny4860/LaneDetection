# -*- coding: utf-8 -*-
import imageio
imageio.plugins.ffmpeg.download()


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from detector.framework import ImageFramework

def process_image(image):
    img_framework = ImageFramework()
    return img_framework.run(image)
 
white_output = 'project_video_result.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)




