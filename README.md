A quick-dirty model to solve the speed prediction challenge assigned by comma.ai/. Goal is to predict vehicle speed from dvr footage. 3D Convolutional Network is used with first dimension being time-duration.  The trained network makes reasonable prediction on validation set with MSE ~2.4.

**The final mean squared error on validation set is 2.4!**

Note that this is the result when the network is purely trained on comma.ai train.mp4. Most of the test-time error occurs at the stopping of vehicles. 

Take a look at a similar attempt at https://github.com/JonathanCMitchell/speedChallenge

Before running the code, change data_folder in speed.py to where your data folder is located, which should contain train.mp4, test.mp4, and train.txt.

# Usage

1. Download the weights pretrained on the Sports-1M dataset [https://goo.gl/tsEUo2]

2. python speed.py --prepdata

3. python speed.py --mode=train

4. python speed.py --mode=test 

# Todo list

1. Use shorter look-back time window to allow more fine-grained speed prediction. At the moment, the network looks 16 frames back in time and predicts the average speed during this period.
2. Use semantic segmentation to cut off moving objects. Moving objects such as vehicles confuses the speed prediction w.r.t. the road. More data with diverse conditions may also be able to clear this issue.
