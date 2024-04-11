Certified Robustness on Visual Graph Matching via Searching Optimal Smoothing Range

This repository contains code and trained models for the paper "Certified Robustness on Visual Graph Matching via Searching Optimal Smoothing Range".

1. Outline

The contents of this repository are as follows:
    ----data/ contains the raw data of PascalVOC-Keypoint. We should download the dataset first.
    ----experiments/ contains the parameter settings of matching methods.
    ----models/ cotains the basic algorithms of matching methods.
    ----figure/ contains plotted figures of certified results.
    ----scr/ contains various modules of gm solvers.
    ----keypoint_trained_model/ contains the trained models under keypoint position perturbations.
    ----pixel_trained_model/ contains the trained models under pixel perturbations.
    ----marginal_radii_under_keypoint/ contains the marginal radii results under keypoint position perturbations.
    ----result_sampling_evaluation/ contains the sampling_simplification results for perturbation pairs.
  
2. Get Started

(1) The basic running environment is the similar as in https://github.com/Thinklab-SJTU/ThinkMatch. We recommend to use Docker.
     Get the recommended docker image by docker pull runzhongwang/thinkmatch:torch1.6.0-cuda10.1-cudnn7-pyg1.6.3-pygmtools0.2.4

(2) Install other dependencies:
     conda install pandas statsmodels matplotlib seaborn

(3) Download PascalVOC-Keypoint.
    Download VOC2011 dataset from http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html and make sure it looks like data/PascalVOC/TrainVal/VOCdevkit/VOC2011
    Download keypoint annotation for VOC2011 from https://drive.google.com/file/d/1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR/view  and make sure it looks like data/PascalVOC/annotations
    The train/test split is available in data/PascalVOC/voc2011_pairs.npz. This file must be added manually.
    Use a similar approach to obtain the willow dataset.

3. Scripts

(1) The program train_under_keypoint_position_perturbation.py trains the model using data augmentation and the regularization term when perturbing keypoint positions.
   python train_under_keypoint_position_perturbation.py --cfg experiments/vgg16_ngmv2_voc.yaml
   We can change --cfg to certify different solvers. Similarly, we can obtain the trained models under pixel perturbations using train_under_pixel_perturbation.py.

(2) The program optimization.py obtains the optimal smoothing range for trained models.
    python  optimization.py

(3) The program certify_under_keypoint_position_perturbation.py calculates maginal radii when perturbing keypoint positions and save the results in "marginal_radii_under_keypoint" . For example:
    python certify_under_keypoint_position_perturbation.py --cfg experiments/vgg16_ngmv2_voc.yaml
    We can change --cfg to certify different solvers. Similarly, we can obtain the maginal radi under pixel perturbations using certify_under_pixel_perturbation.py.

(4) To analyze the certification, if we use sampling evaluation, we can run:
    python sampling_simplification.py
    python analyze_sample_evaluation.py
    We can obtain the probability that the samples belong to the certified space.

(5) To analyze the certification, if we use marginal radii evaluation, we can run:
    python analyze_marginal radii.py
    We can obtain figures for RS-GM and CR-OSRS in "figure". And we can run the ACR.py to obtain ACR results.

(6) We can run the following file to evaluate the parameters:
    python analyze_different_sigma.py
    And we can change the parameters in file to evaluate different parameters.




