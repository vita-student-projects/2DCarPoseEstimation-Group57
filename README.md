# 2DCarPoseEstimation-Group57
2DCarEstimation with HR Former in openpifpaf backbone with Carfusion dataset


# Introduction

This project aims to use machine learning and specifically state-of-the-art models in order to estimate vehicle's 2D positions on images and further on videos. This problem is very important in the autonomous vehicles' world and is therefore crucial in general.
A 2D pose estimation model takes as input either and image or a video consisting of frames and outputs the pixel positions of each keypoint on the image. This can be used to visualize the frame of a car or further to transform these keypooints into 3D keypoints and model for a fully developped environment.

Two of the main models that have been developped and that we decided to focus on are OpenPifPaf and Occlusion-Net. The first can be used for various purposes such as human pose estimation but in this case also vehicle pose estimation. The second one is specifically used for vehicles and one of its main uses is to first find the visible keypoints of a car and then from these keypoints, find the occluded ones.

Our main goal in this project was to merge these two models such that anything inputted in OpenPifPaf would result in visible keypoints that would then be used in the second model to find the full position of the vehicles.

Sadly, working with Occlusion-Net has proven to be way more difficult than we thought it would be, and therefore we had to resize our project and end it before the full implementation.

# Contribution Overview

We have integrated the use of the OpenPifPaf code in our work, enhancing it by incorporating a High Resolution (HR) Transformer into the backbone of our algorithm. Further, we have modified the dataset to ensure a comprehensive representation of vehicles. This includes not just the visible parts, but also those parts that are typically obscured, thereby enabling a more accurate and holistic understanding of the subject.


![Shema](https://media.arxiv-vanity.com/render-output/6205766/images/model.png)


HRFormer is an architecture in machine learning that has been created recently and consists of a network branching out into multiple parallel lines. It is said to reduce both computational cost and memory and results in good performances when tested on the COCO dataset.
It can and has been used for human pose estimation, segmentation and classification.

![HRFormer](https://raw.githubusercontent.com/HRNet/HRFormer/main/cls/figures/HRFormer.png)

Our first objective to add the Occlusion-Net model with a trifocal tensor loss was too complicated to implement, due to the multiple interactions between each part of the model that made the modification of a part very complicated for us.


In the preparation of our goal, we implemented a CarFusion plugin in OpenPifPaf and an additional Backbone to test our Occlusion-Net model, which sadly had to be dropped out of the project.

![OcclusionNet](https://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/cvpr2019/images/title_1.jpg)

# Experimental Setup
`What are the experiments you conducted? What are the evaluation
metrics?`

We decided to find the best model backbone to adapt with the new dataset that we decided to use (see info below) so we took our attention to the shufflenet of openpifpaf and the hr-former. We made some training run with this two backbones and we have evalutated with a testset to find the best one to continue with. We used final weights of this different model as a checkpoint to start training our models, like that the model don't start with random weights and it can perform the behaviour easier and it just need to understand our heads.

To know wich model is the best, we used some metrics:

The keypoint detection task COCO is scored like an object detection task, with average precision (AP) and average recall (AR)

- Average Precision (AP):
AP measures the accuracy and precision of the model's predictions for a specific category, such as a keypoint or an object class. It calculates the precision at different levels of recall by varying a threshold for considering a prediction as a true positive. AP ranges from 0 to 1, where a higher value indicates better performance. 
- Average Recall (AR):
AR measures the model's ability to detect instances of a specific category at various levels of precision. It calculates the recall at different levels of precision by varying a threshold for considering a prediction as a true positive.

AP and AR are in range from 0 to 1, where a higher value indicates better performance.


 - We used also the Mean Pixel Error and Detection Rate to choose the best model.



# Dataset

We use the dataset used in Occlusion-Net, CarFusion, that consists of 5 videos of cross street intersection with cars running through it and from which frames have been turned into jpg images and annotated.

This results in about 50'000 images and about twice as many cars visible.
Theses images have been annotated with 14 keypoints, some being visible and also the others being occluded.

The keypoints represent the 4 wheels, 4 lights, 4 roof corners of the car and the exhaust and center point. These two last points, although they are annotated, are not shown in the skeletons of the cars found, such as seen below:

![14 keypoints](https://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/cvpr2019/images/title/3.png)

In order to use this dataset, the firt step is to download it and then to transfer it to COCO format, which will be easier to work with.

## 1. Download dataset

 - First we need to download the dataset and unzip it.

The links to the data are:

Craig Sequence

Part1: https://drive.google.com/opend=1L2azKaYabCQ0YWwGTnN0_uhNgnlACigM

Part2: https://drive.google.com/open?id=1Z15rX038FyRW62DOPEJW16bZZIj8LPC0

Fifth Sequence

Part1: https://drive.google.com/open?id=1W0Ty7vZbnAGyvf0RNeVXcNJV6e7kcqzZ

Part2: https://drive.google.com/open?id=1T253FkwEfKEJyegHNztRPQX_bSjZUznf

Morewood Sequence

Part1:https://drive.google.com/open?id=1uuDYPidIztuOHbDbP2hQv_ocGAnWPbGJ

Part2: https://drive.google.com/open?id=1R3xpUKjHPFOlOj8SZIV9hmtndFMw95iX

Butler Sequence

Part1: https://drive.google.com/open?id=1goDsHDSU5dkJFoy0phTFD2brXf4Ou6sL

Part2: https://drive.google.com/open?id=1AOFQrLQTkNs1djnfI7y-ZeeZQeqnHNQK

Penn Sequence

Part1: https://drive.google.com/open?id=1dYqJtokx1pMEtWH8XK9b_R481jv2ypn2

Part2: https://drive.google.com/open?id=12vSzUh_e3oMYVKewUeEzn9Gv3P_wnGYi


This data can be very memory-heavy so we advise you not to download and use all of it unless it is required.

Once downloaded, the data should be moved into src/openpifpaf/dataCarFusion/[train, test] as needed.

The full dataset weighs 32 Gb and should be split into two: the Craig, Fifth, Morewood and Butler should be used as training set wheras the Penn dataset should be used as testing.
Other ways to split the dataset can be to use only the first half of each dataset and to split them the same way, this results in 16 Gb used.\
You can also only use the first half of the Craig dataset as training and the first half of Butler as testing for a light dataset of 5 Gb.

The final folder should look as follows:

```text
└─dataCarFusion
    └─train
        └─car_craig1
            └───images
                01_0000.jpg
                01_0001.jpg
                ...   
            └───bb
                01_0000.txt
                01_0001.txt
                ...
            └───gt
                01_0000.txt   
                01_0001.txt
                ...
    └─test
        └─car_penn1
            └───images
                01_0000.jpg
                01_0001.jpg
                ...   
            └───bb
                01_0000.txt
                01_0001.txt
                ...
            └───gt
                01_0000.txt   
                01_0001.txt
                ...                
```


## 2. Transform data

Before going further, we advise the user to create a virtual environment with Python 3.7, as it will be used for the data transformation and the project management. We let the user do this freely using either Conda or VirualEnvironment.

Before preprocessing the data, the user install the files as a package using pip with the following command:
```
pip install -e .
```
The packages needed should be installed, but be aware that for some specific uses, for example a backbone that is outside of the scope of this project, there could be extra requirements.

Then, we must transform the dataset so that it uses the Coco format. We have created a file called dataset.py that does the transformation of the images in the dataCarFusion folder.\
This code relies partly on the following github : https://github.com/dineshreddy91/carfusion_to_coco.

This file is located in /src/openpifpaf/, as are any other useful files in this project.


```
pip install -r requirements.txt
python dataset.py --path_dir dataCarFusion/train/ --output_dir dataCarFusion/annotations/ --output_filename car_keypoints_train.json
python dataset.py --path_dir dataCarFusion/test/ --output_dir dataCarFusion/annotations/ --output_filename car_keypoints_test.json
```

You can verify that the code worked correctly by checking if both test and train annotations have been created under src/openpiafpaf/dataCarFusion/annotations/.

# Code

`ENCORE A FAIRE : EXPLIQUER TRAIN ET LES OPTIONS, COMMENT UTILISER, PAREIL POUR INFERENCE`

Other important files that can be used are train.py and inference.py, both can be find in /src/openpifpaf/.

The repository includes the following scripts:

- `dataset.py`: Implements the dataset class for loading and preprocessing the dataset.
- `train.py`: Provides the training script for the model using the dataset.
- `inference.py`: Enables inference on new images using the trained model.

The dataset.py has already been explained above.

The training part of the model can be done with several base networks, datasets and checkpoints.

One example of such use can be seen here:
```sh
python -m openpifpaf.train --dataset carfusionkp --checkpoint=shufflenetv2k16-apollo-24 --carfusionkp-square-edge=513 --lr=0.00002 --momentum=0.95  --b-scale=5.0 --epochs=350 --lr-decay 320 340 --lr-decay-epochs=10  --weight-decay=1e-5 --weight-decay=1e-5  --val-interval 10 --loader-workers 16 --carfusionkp-upsample 2 --carfusionkp-bmin 2 --batch-size 2
```
This perticular command uses our implemented CarFusion dataset with the preexisting ApolloCar 24kps shufflenet checkpoint.

A complete list of general training options and checkpoints can be found here: https://openpifpaf.github.io/cli_help.html

Similarly to other datasets, CarFusion options can be found here:
```
--carfusionkp-train-annotations         Path to train annotations
--carfusionkp-val-annotations           Path to val/test annotations
--carfusionkp-train-image-dir           Path to train images directory
--carfusionkp-val-image-dir             Path to val/test images directory
--carfusionkp-square-edge               Square edge of input images
--carfusionkp-extended-scale            Augment with extended scale range
--carfusionkp-orientation-invariant     Augment with random orientations
--carfusionkp-blur                      Augment with blur
--carfusionkp-no-augmentation           Disable augmentations
--carfusionkp-rescale-images            Rescale factor for images
--carfusionkp-upsample                  Head upsample stride
--carfusionkp-min-kp-anns               Minimum number of keypoint annotations
--carfusionkp-bmin                      bmin
```

Most of these options have default values and do not need to be set if otherwise needed.


# Results
`Qualitative and Quantitative results of your experiments.`

## RESULTS with HR-FORMER after 30 epoch:

Evaluate annotation type *keypoints*
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.050
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.125
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.007
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.050
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.075
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.160
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.031
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.064

Final Results: 
Mean Pixel Error [scaled] : 3.432623 
Detection Rate [scaled]: 49.254017 %
```


<p align="center">
<img src="images/image_hrformer.jpeg", width="900">
<br>
<sup>Result of HR_former train with 30 epoch</sup>
</p>


## RESULTS with SHUFFLENET after 30 epoch:

Evaluate annotation type *keypoints*

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.113
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.265
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.018
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.115
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.159
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.343
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.070
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.153

Mean Pixel Error [scaled] : 3.126907 
Detection Rate [scaled]: 62.110065 %
```


<p align="center">
<img src="images/image_shufflnetv1.jpeg", width="900">
<br>
<sup>Result of shufflnet train with 30 epoch</sup>
</p>

As seen in the results above, the shufflenet is really better to detect the car 62%>49% but is also better twice better in AP and AR.
Therefore we decide to keep the shufflenet and to continue to train it.

## RESULTS with SHUFFLENET after 60 epoch

Evaluate annotation type *keypoints*

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.128
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.271
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.063
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.130
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.190
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.381
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.130
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.187

Mean Pixel Error [scaled] : 2.939359 
Detection Rate [scaled]: 60.617481 
```


<p align="center">
<img src="images/image_shufflenetv2.jpeg", width="900">
<br>
<sup>Result of shufflnet train with 60 epoch</sup>
</p>


 # Analyse of results

<p align="center">
<img src="images/image_shufflenet_original.jpeg", width="900">
<img src="images/shufflnet_v2.jpeg", width="900">
<br>
<sup>Result of shufflnet-24keypoints (1st image) and our model (2nd image)</sup>
</p>

 There is some interesting results, it seems that our model became less powerfull to detect cars but it became very good in precision for long distance vehicule. From the model trained with 30 epoch, we see that the Mean Pixel Error was higher that our new model but the detection rate was better. The Mean Pixel Error is 0 when the model don't detect any car, it mean that the model tends to prefer to not detect a car if it's not very sure about it. It prefer to not showing than showing a bad prediction and we could say that our final model is not better but if we check the average precision (AP) and the average recall (AR) everything is better and the precision and the recall above 75% double.

 Compare to the original model from openpifpaf, we have some much less cars detected but we have most of the time better result in long distance detection in cross road intersection, that's because our dataset is essentially cross road intersection.  

# Conclusion
`here a short conclusion`



# How to run and test with an image

- Put an image into the src folder and call it voiture.jpg

go into src with the terminal and do:
```sh
python3 -m openpifpaf.predict \
  --checkpoint outputs/shufflenetv2k16-230524-164104-carfusionkp-a09acb3d.pkl \
  --save-all --image-output all-images/overfitted.jpeg \
  voiture.jpg
```
Example result:
![example image cars](https://raw.githubusercontent.com/openpifpaf/openpifpaf/main/docs/peterbourg.jpg.predictions.jpeg)
