import cv2
import new_models as models
import argparse
import tensorflow as tf
import numpy as np
import os
from os.path import isfile, join
import natsort
# class mine(models.BaseSuperResolutionModel):

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    #files.sort(key = lambda x: int(x[5:-4]))
    files = natsort.natsorted(files)
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def main(ab):
    pathIn= './final1/final0/'
    pathOut = 'output3.avi'
    fps = 30.0
    scale_factor = int(2)
    suffix = "scaled"
    patch_size = int(8)
       
    vidcap = cv2.VideoCapture(ab)
    success,image = vidcap.read()
    count = 0
    model = models.ResNetSR(scale_factor)
    mod = model.initial("final0/frame%d.jpg" % count, save_intermediate=False, mode="fast", patch_size=patch_size, suffix=suffix)
    while success:
        cv2.imwrite("final0/frame%d.jpg" % count, image)     # save frame as JPEG file     
        model.upscale(mod, "final0/frame%d.jpg" % count, save_intermediate=False, mode="fast", patch_size=patch_size, suffix=suffix) 
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    convert_frames_to_video(pathIn, pathOut, fps)
if __name__=="__main__":
    main(ab)
