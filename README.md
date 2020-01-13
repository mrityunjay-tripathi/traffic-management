<style>
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 70%;
}
</style>

<h1 align = "center" font = "">Traffic Management - The Unconventional Way</h1>
<img src = "../traffic_management/data/custom/image1.jpg" class = "center">

<p align = "center">This repository is a project under <b>Meghalaya Police Hackathon 2020</b>.</p>

## Overview
Recently, Meghalaya Home Minister James P.K. Sangma told the Assembly that around 1.7 lakh vehicles pass through Shillong during peak hours and around 93,000 on a normal day. According to a study by the Boston Consulting Group, India’s biggest cities are losing approximately US$22 billion (Rs. 1,62,200 crore) annually due to traffic congestion. Due to narrow roads and a large number of vehicles per capita, frequent traffic congestion occurs in Shillong. It leads to wastage of time for commuters, employees, students. Blocked traffic may interfere with the passage of emergency vehicles (eg. Ambulance, Army/Police Vehicle, etc.). Higher chance of collisions due to tight spacing and constant stopping-and-going.

## Objectives
* Detecting vehicles and their License Plates using road-side cameras.
* Analyzing parking areas and suggesting parking slots to new vehicles using mobile app.  

## Prerequisites
#### Environment
* Python >= 3.6.0
* Pytorch >= 1.3.0

#### Get code
```
git clone https://github.com/mrityunjay-tripathi/traffic-management.git
cd traffic_management
pip3 install --user -r requirements.txt 
```
#### Download required dataset
```
cd data/
bash get_traffic_data.sh
```

## Training
##### Download pre-trained weights.
```
cd weights/
bash lpr_weights.sh
bash ps_weights.sh
```
##### Modify training parameters
1. Review config file ```config/params.cfg```
3. Adjust your GPU device. see parallels.   
4. Adjust other parameters.   
##### Start training
```
python train.py
```
##### Optional: Visualize training
```
tensorboard --logdir=PATH_WHERE_TENSORBOARD_LOGS_ARE_SAVED
```

<img class = "center" src="data/custom/loss_curve.png">

## Test
Put your test data in ```data/samples/test/``` then run the following command.
```
python test.py
```

## Start to use
```
python detect.py
```

## Acknowledgement
* [YOLO-v3](https://pjreddie.com/darknet/yolo/)

## Author
* [Mrityunjay Tripathi](https://www.linkedin.com/in/mrityunjay-tripathi-89a243168/)