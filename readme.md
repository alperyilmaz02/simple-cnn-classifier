Simple CNN Image Classifier with ROS

This repository contains a simple CNN-based image classifier integrated with a ROS package (aesk_net).
It supports both standalone training/testing in Python and real-time inference via ROS nodes.

🚀 Installation
1. Clone the repository

2. Install the Python package

Inside your repo run this where setup.py is located:

pip install -e .

3. (Optional) Build ROS workspace for testing real time 
cd test_with_ros
catkin_make
source ./devel/setup.bash

🧠 Usage

2. Training
python3 train.py --train_set path/to/trainset --val_set path/to/valset --batch_size 16 --epoch 40 --img_size 16 

3. Testing
cd test_with_ros
roslaunch aesk_net aeskNet.launch

