# video-super-resolution
Frame by frame super resolution of low quality video to high quality video
If you wish to train the network on your own data set, follow these steps (Performance may vary) :
[1] Save all of your input images of any size in the "input_images" folder
[2] Run img_utils.py function, transform_images(input_path, scale_factor). By default, input_path is "input_images" path.
[3] Execute ftests.py to begin training. GPU is recommended, although if small number of images are provided then GPU may not be required.
