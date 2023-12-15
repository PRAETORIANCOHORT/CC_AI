# Predicting_5_year_Recurrence_Risk_in_Colorectal_Cancer_a_Histology-Based_Deep_Learning_Approach

## Model background

This is a deep learning model based on WSIs to predict the 5-year recurrence risk of colorectal cancer patients. There are two parts of the code:
1. The main scripts part: our model
2. The CAM part: show our CAM visualization

## Workflow of the model

1. After we input a WSI of the patient, we need to cut the WSIs into 512*512 pixel patches under 10X
2. Running MILTrain.py in main scripts to train the model. In the training process, the model will sample clusters of patches with spatial relationship, in order to 'zoom up' the view of the model
3. Use Multiple-instance learning (MIL) to train our model on patch-clusters.
4. Test the model with MILTest.py
5. Get hotmap visualization with script MILHotmap_df.py
6. Get Class Activation Map (CAM) with script CAM_all.py

## Training arguments

1. path: The path of patches
2. model_id: The name of model
3. epochs: The number of epochs, default=100
4. mag: Magnification of patches, default='10'
5. lr: Initial learning rate, default=0.0002
6. momentum: Momentum, default=0.90
7. weight_decay: Weight_decay, default=1e-4
8. lrdrop: lrdrop, default=50
9. padding: The number of training clusters, default=4
10. test_limit: The number of testing clusters, default=50
11. extd: The number of patches in a cluster other than the central patch, default=11
12. device: The ID of GPU device, default='0,1,2,3,4,5,6,7'
13. comment: Comment files, default='comment'
14. model: Feature extractor model name, default='inceptionv3'

   
   


