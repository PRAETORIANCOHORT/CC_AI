# CC_AI

## Model background

This is a deep learning model based on WSIs to predict the 5-year recurrence risk of colorectal cancer patients
There are two parts of the code:
1. The main scripts part: our model
2. The CAM part: show our CAM visualization

## Workflow of the model

1. After we input a WSI of the patient, we need to cut the WSIs into 512*512 pixel patches under 10X as follow:
![patches images](https://github.com/PRAETORIANCOHORT/CC_AI/tree/main/images/img2.png "patches images")
2. Running MILTrain.py in main scripts to train the model. In the training process, the model will sample clusters of patches with spatial relationship, in order to 'zoom up' the view of the model:
![patch-clusters images](https://github.com/PRAETORIANCOHORT/CC_AI/tree/main/images/img1.png "patch-clusters images")
3. Use Multiple-instance learning (MIL) to train our model on patch-clusters.
4. Test the model with MILTest.py
5. Get hotmap visualization with script MILHotmap_df.py
6. Get Class Activation Map (CAM) with script CAM_all.py

   
   


