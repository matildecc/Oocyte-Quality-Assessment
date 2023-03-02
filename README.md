# Oocyte-Quality-Assessment

**Abstract**:


The success rate of bovine in vitro embryo reproduction is low and highly dependent on the oocyte quality. The selection of the oocyte to be fertilized is done by the embryologists’
visual examination of oocytes. It is time-consuming, subjective, and inconsistent between specialists in the area. In this repository, a semi-automatic solution is proposed to score the
quality of an immature oocyte. It consists of a deep learning model to classify oocyte competence. The model was trained and tested with real data, composed of images of immature oocytes
and their label of whether they developed into blastocysts after fertilization. To the best of our knowledge, automated bovine oocyte classification was not attempted before, but experimental
results show that our proposed solution is more robust and objective than specialists’ visual assessment and comparable with other works on human oocytes.


**How to run:**

1. Download dataset from the drive´s shared link of the .txt file 
2. Create a folder with the name 'original images' inside the folder 'database - immature' and save the downloaded images.
3. Create 2 folders with the name 'images with center' and 'center cropped images' inside the folder 'database - immature'
4. Run 'ManualCenterLocation.py' and select the center of all the images. 
5. Run 'CenterCropping.py' 

By this time, you should have your pre-processing complete (oocyte center location + center cropping)

4. Run 'OocyteCompetenceAssessment.py´ to train the model with the dataset (please pay attention to the comments)
5. For testing, you should comment the training section of the py file and uncomment the testing section. 


**Alternative:** 
To an extra pre-processing, you can run the file 'Reinhard.py' after center cropping (5.), to normalize the dataset color. This was just a trial, there was no evident benefit of this step.


**FastRunning:** 
Instead of 1.- 4., you can download from the drive's shared link the images with the label of their oocyte center coordinates. Do not forget to save this images in the folder 'images with center' and create an empty one named ''center cropped images'. 
Afterwards, you just need to perform step 5.



