# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

def video_to_image(mp4_file='project_video.mp4'):
    vidcap = cv2.VideoCapture(mp4_file)
    
    while True:
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            yield image
        else:
            break

video = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1280,720))

images = []
for i, img in enumerate(video_to_image('project_video.mp4')):
    # process lane finding
    # save images
    video.write(img)
video.release()
cv2.destroyAllWindows()

