# Action-Recognition-using-ResNet-Kinetics
The script of recognizing an action in 10 second video using pretrained ResNet on Kinetics dataset

In action recognition we are trying to know what the action is in the clip we have.
There are many pretrained model that makes things easy for us. Here I am using
ResNet to make it work. 

For action recognition providing a singal or static frame to the model wouldn't help
us get the desire result. We have to provide the model with a set of frame which should 
be clearly defining the action.
# Requirements
  - OpenCV
  
# Downloads
Download the ResNet pretrained Model from the link
https://drive.google.com/file/d/1hFcRwBSZZlajGIZspqjPDfUrbVKEIqzH/view?usp=sharing

#If you want to have a look at dataset 
https://deepmind.com/research/open-source/kinetics

The file action_recognition.txt list all the action that can exists according to 
kinetics700 dataset and using this we will predict the action.

# How it works
Simply run the code.py and if you want to change the video for test you can do that while changing the path in the code
After loading all the desire files we will start on the core code.
We will cluster the set of frames which we will provide to the model. We can variate
the clustering duration. After providing the sets of frame to the model, model will
predict the label for all the individual frame which we can annotate on the frame 
and output it.


