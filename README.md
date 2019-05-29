# Fire-Detection
Detect fire in surveillance videos with flame and smoke detection

Fire Detection has been done by creating Flame and Smoke detecting models using TensorFlow Object Detection API ( https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html ).

Datasets used:

[1] FireSense (https://zenodo.org/record/836749#.XFkUHlwzbIU)

[2] FurgFire (https://github.com/steffensbola/furg-fire-dataset)

[3] http://smoke.ustc.edu.cn/datasets.htm

[4] https://github.com/cair/Fire-Detection-Image-Dataset


Frames were captured from video datasets and compiled with other images to create flame and smoke datasets. They were annotated using labelImg.

After models were created they were tested on images and videos. First images are checked for flame and then for smoke.
They were tested on real time footage by simulation of CCTV camera by smartphone camera using DroidCam(https://play.google.com/store/apps/details?id=com.dev47apps.droidcam&hl=en_IN)
