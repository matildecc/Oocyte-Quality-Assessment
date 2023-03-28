# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:30:17 2022

3rd Script - oocyte quality classifier 
It includes the data organization, the model, the training and the testing.  

@author: Matilde
"""

#%% imported libraries

import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import cv2

#%% random seeds 
seed = 0
torch.manual_seed(seed)

"FUNCTIONS"
#%% ImageDataset class

class ImageDataset(Dataset):
  def __init__(self,df,img_folder,transform):
    self.df=df
    self.transform=transform
    self.img_folder=img_folder
    self.image_names = self.df[:]['image_name']
    self.labels = torch.Tensor(np.array(self.df[:]['Blastocyst Day8']).astype('uint8'))
   
#The __len__ function returns the number of samples in our dataset.
  def __len__(self):
    return len(self.image_names)
 
  def __getitem__(self,index):
    image=Image.open(self.img_folder+str(self.image_names.iloc[index])).convert('RGB')
    image=self.transform(image)
    targets=self.labels[index]
     
    return image, targets
#%% Model Arquitecure 

class Net(nn.Module): 
    def __init__(self):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(p=0.6),
            
            nn.Conv2d(32,64, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(p=0.5),
            
            nn.Conv2d(64, 128, 3,  padding='same'),
            nn.ReLU()
            )
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*128 , 128), 
            nn.ReLU(),
            nn. Linear(128,1)
            )
        
        self.gradient = None
    
    #important for gradCam    
    def activations_hook(self, grad):
        self.gradient = grad

    def forward(self, x):
        x  = self.feature_extractor(x)
        
        #important for gradCam  
        h = x.register_hook(self.activations_hook)
        
        x = self.maxpool(x)
        x = self.classifier(x)
        
        return x
    
    def get_activation_gradients(self):
        return self.gradient
    
    def get_activation(self, x):        
        return self.feature_extractor(x)
        
        
#%% Epoch method 
# outputs: loss_per_epoch, accuracy, TPR, TNR, AUC
def epoch_iter(dataloader, model, loss_fn, device, optimizer=None, is_train=True):
    
    if is_train:
      assert optimizer is not None, "When training, please provide an optimizer."
      
    num_batches = len(dataloader)

    if is_train:
      model.train() # put model in train mode
    else:
      model.eval()

    total_loss = 0.0
    preds = []
    labels = []
    

    for batch, (X, y) in enumerate(tqdm(dataloader)):
        #y = y.type(torch.LongTensor) 
        X, y = X.to(device), y.to(device)
       

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.reshape(-1,1))
        #print(torch.sigmoid(pred))

        if is_train:
          # Backpropagation
          optimizer.zero_grad() #sets the gradient to zero, before computed to not accumlate from the previous iter
          loss.backward() #The gradients are computed 
          optimizer.step() #performs a parameter update based on the current gradient
  
          
        # Save training metrics
        total_loss += loss.item() # IMPORTANT: call .item() to obtain the value of the loss WITHOUT the computational graph attached
                        
        final_pred = (torch.sigmoid(pred)>0.5).int()
        preds.extend(final_pred.cpu().numpy())
        labels.extend(y.cpu().numpy())
          
    confusion_matrix = metrics.confusion_matrix(labels, preds)
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]

    return total_loss / num_batches, accuracy_score(labels, preds), recall_score(labels, preds), TN/(TN+FP), roc_auc_score(labels, preds)
#%% auxiliary function to plot the loss and accuracy during training
def plotTrainingHistory(train_history, val_history, num_epochs):
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.scatter(range(1,num_epochs+1),train_history['loss'], label='train', color = 'black')
    plt.scatter(range(1,num_epochs+1),val_history['loss'], label='val', color = 'red')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.title('Classification accuracy')
    plt.scatter(range(1,num_epochs+1),train_history['accuracy'], label='train', color = 'black')
    plt.scatter(range(1,num_epochs+1),val_history['accuracy'], label='val',  color = 'red')

    plt.tight_layout()
    plt.show()
    
#%% Training function
#Saves the following models : 1) best val loss, 2) best val TPR, 3) best val TNR, 4) Best val AUC 5) the last one
#outputs the train and val history (loss, accuracy, TPR, TNR)

def train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, num_epochs, model_name, device):
  train_history = {'loss': [], 'accuracy': [], 'TPR': [], 'TNR': []}
  val_history = {'loss': [], 'accuracy': [] , 'TPR': [], 'TNR': []}
  best_val_loss = np.inf
  best_val_TPR = 0.0
  best_val_TNR = 0.0
  best_val_acc = 0.0
  best_val_auc = 0.0
  train_loss = 0.0
  print("Start training...")
  
  for t in range(num_epochs):
         
    print(f"\nEpoch {t+1}")
    train_loss, train_acc, train_TPR, train_TNR, train_auc = epoch_iter(train_dataloader, model, loss_fn, device, optimizer)
    print(f"Train loss: {train_loss:.3f} \t Train accuracy: {train_acc:.3f} \t TPR: {train_TPR:.3f} \t TNR: {train_TNR:.3f} \t AUC: {train_auc:.3f}")
    
    val_loss, val_acc, val_TPR, val_TNR, val_auc = epoch_iter(validation_dataloader, model, loss_fn,device, is_train=False)
    print(f"Val loss: {val_loss:.3f} \t Val accuracy: {val_acc:.3f} \t TPR: {val_TPR:.3f} \t TNR: {val_TNR:.3f} \t AUC: {val_auc:.3f}")

    # save model when val loss improves
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
      torch.save(save_dict, model_name + '_best_val_loss.pth')
     
     
    if best_val_TPR < val_TPR and val_TNR != 0:
      best_val_TPR = val_TPR
      save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
      torch.save(save_dict, model_name + '_best_val_TPR.pth')
      
    if best_val_TNR < val_TNR and val_TPR!= 0:
      best_val_TNR = val_TNR
      save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
      torch.save(save_dict, model_name + '_best_val_TNR.pth')
      
    if t%10 == 0:
        save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
        torch.save(save_dict, model_name + 'model_after_' + str(t) + '_epochs.pth')
      
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
        torch.save(save_dict, model_name + '_best_val_acc.pth')
        
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
        torch.save(save_dict, model_name + '_best_val_auc.pth')
        
        
    # save latest model
    save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
    torch.save(save_dict, model_name + '_latest_model.pth')

    # save training history for plotting purposes
    train_history["loss"].append(train_loss)
    train_history["accuracy"].append(train_acc)
    train_history["TPR"].append(train_TPR)
    train_history["TNR"].append(train_TNR)
    
    

    val_history["loss"].append(val_loss)
    val_history["accuracy"].append(val_acc)
    val_history["TPR"].append(val_TPR)
    val_history["TNR"].append(val_TNR)
      
  print("Finished")
  return train_history, val_history

#%% auxiliary function to plot the ROC curve 
from sklearn.metrics import roc_curve, classification_report
from sklearn.metrics import RocCurveDisplay

def plot_sklearn_roc_curve(y_real, y_pred):
    fpr, tpr, _ = roc_curve(y_real, y_pred)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    roc_display.figure_.set_size_inches(5,5)
    plt.plot([0, 1], [0, 1], color = 'g')

#%% GradCAM section 

#Check the slides of presentation 5th meeting, slide 12 to understande the labels 

from torch.nn.modules import activation
def get_gradcam(model, image, label, size):
    
    label.backward()
    gradients = model.get_activation_gradients()
    print(gradients.shape)
    pooled_gradients = torch.mean(gradients, dim = [0,2,3]) #a1, a2, a3,...., ak
    print(pooled_gradients.shape)
    activations = model.get_activation(image).detach() #A1, A1, A3, ....., AK #
  
    for i in range(activations.shape[1]):
      activations[:,i,:,:] *= pooled_gradients[i]
  
    heatmap = torch.mean(activations, dim = 1).squeeze().cpu()
    heatmap = nn.ReLU()(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(),(size, size))
  
    return heatmap


def plot_heatmap(image, pred, heatmap, label):

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(20,20), ncols=3)

    classes = ['Became blastocyst']
    ps = torch.sigmoid(pred[0]).cpu().detach().numpy()
    ax1.imshow(image)

    ax2.barh(classes, ps[0])
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Predicted Class ' + str(round(ps[0], 2)) + ' and True Class : ' + label)
    ax2.set_xlim(0, 1.1)


    ax3.imshow(image)
    ax3.imshow(heatmap, cmap='magma', alpha=0.7)
    

"MAIN"
#%% Data Organization

source_path =  "./database - immature/center cropped images/"
df = pd.read_excel('./database - immature/images_info.xlsx', dtype = str)

#complete the df with oocyte center information 
for path in os.listdir(source_path):   
    oocyte = (path.rpartition('-')[2][:-4])
    if not pd.isna((path)):
        df.loc[df['oocyte n°'] == oocyte,'image_name'] = path
     
#select the images accpeted by AR
new_df = df[(df["Accepted_AR (Y/N)"] == 'Y')]

new_df_class_1 = new_df[(new_df['Blastocyst Day8'] == '1')] #competent oocytes
new_df_class_0 = new_df[new_df['Blastocyst Day8'] == '0'] #non-competent oocytes

#Training : 70% of images
num_class_1_train = round(0.70*len(new_df_class_1))
num_class_0_train = round(0.70*len(new_df_class_0))
new_df_class_0_train = new_df_class_0.sample(n=num_class_0_train, random_state=seed)
new_df_class_1_train = new_df_class_1.sample(n=num_class_1_train, random_state= seed)


#Validation: 20% of images
num_class_1_val =  round(0.20*len(new_df_class_1))
#Balance 60%/40%
new_df_class_1_val = new_df_class_1[~new_df_class_1.isin(new_df_class_1_train)].dropna(subset = ['oocyte n°']).sample(n = num_class_1_val, random_state= seed)
new_df_class_0_val = new_df_class_0[~new_df_class_0.isin(new_df_class_0_train)].dropna(subset = ['oocyte n°']).sample(n = round((0.6*num_class_1_val)/0.4), random_state= seed)


# Test set: 10% of images
new_df_class_1_test = new_df_class_1[~new_df_class_1.isin(new_df_class_1_train) & ~new_df_class_1.isin(new_df_class_1_val)].dropna(subset = ['oocyte n°'])
new_df_class_0_test = new_df_class_0[~new_df_class_0.isin(new_df_class_0_train) & ~new_df_class_0.isin(new_df_class_0_val)].dropna(subset = ['oocyte n°']).sample(n = round((0.6*len(new_df_class_1_test))/0.4), random_state= seed)




#%% Datasets

img_folder =  "./database - immature/center cropped images/"
  

tansforms_augmented = transforms.Compose([
        transforms.ToTensor(), 
        #transforms.Grayscale(),
        transforms.RandomRotation(180, interpolation = transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p = 0.8),
        transforms.RandomVerticalFlip(p = 0.8),
        #transforms.CenterCrop((256,256)) #to avoid the black borders after rotation, in the preprocessing we can change the cropping size for a bigger one
        ])


transforms_ = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Grayscale(),
        ])

#Train Datasets
train_dataset_0_transformed = ImageDataset(new_df_class_0_train,img_folder,tansforms_augmented)
train_dataset_1_transformed = ImageDataset(new_df_class_1_train,img_folder,tansforms_augmented)
train_dataset_0 = ImageDataset(new_df_class_0_train,img_folder,transforms_)
train_dataset_1 = ImageDataset(new_df_class_1_train,img_folder,transforms_)

#adding the augmentation samples
increased_dataset_train = torch.utils.data.ConcatDataset([train_dataset_0_transformed,
                                                          train_dataset_1_transformed,
                                                          train_dataset_1_transformed,
                                                          train_dataset_1_transformed,
                                                          train_dataset_1_transformed,
                                                          train_dataset_1_transformed,
                                                          train_dataset_1_transformed,
                                                          train_dataset_1_transformed,
                                                          train_dataset_1_transformed,
                                                          train_dataset_1_transformed,
                                                          train_dataset_1_transformed,
                                                          train_dataset_1_transformed,
                                                          train_dataset_0_transformed,
                                                          train_dataset_0,
                                                          train_dataset_1
                                                          ])


# Validation Datasets
val_set = pd.concat([new_df_class_0_val, new_df_class_1_val])
val_dataset = ImageDataset(val_set,img_folder,transforms_ )

#Testing Datasets
test_set = pd.concat([new_df_class_0_test, new_df_class_1_test])
test_dataset =  ImageDataset(test_set,img_folder,transforms_ )

#%% Training script
# comment to use the testing script

if __name__ == '__main__':
     
     #Training settings
     IMG_SIZE = 256
     BATCH_SIZE = 64
     NUM_WORKERS = 2
     num_epochs = 150
     LR = 1e-5
        
     #To run in GPU
     device = "cuda" if torch.cuda.is_available() else "cpu"
     print(f"Using {device} device")
     
     
     #Dataloaders
     train_dataloader = DataLoader(
         increased_dataset_train, 
         batch_size=BATCH_SIZE,
         shuffle=True, 
         num_workers=NUM_WORKERS, 
         #pin_memory=True
         
     )
      
     validation_dataloader = DataLoader(
         val_dataset, 
         batch_size=BATCH_SIZE,
         shuffle=False, 
         #num_workers=NUM_WORKERS, 
         #pin_memory=True
     )
         
     
     model = Net()
     model.to(device)
     summary(model, (3,IMG_SIZE, IMG_SIZE), batch_size = BATCH_SIZE)
     
     #More training settings
     loss_fn = nn.BCEWithLogitsLoss()
     optimizer = optim.Adam(model.parameters(), lr=LR)
     
     train_hist, val_hist = train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, num_epochs, model_name="OocyteQuality", device = device)
     plotTrainingHistory(train_hist, val_hist,num_epochs)
     
     
#%% Testing Script
# comment for training

# model = Net()
# model.to(device)
# #Load the best model to feedforward
# model.load_state_dict(torch.load('OocyteQuality_best_val_acc.pth')['model'])
# model.eval() #evaluation mode

# preds = []
# labels = []
# probabilities = []

  
# for image, label in test_dataset:
#     denorm_image = image.permute(1,2,0)
#     image = image.unsqueeze(0).to(device)
   
#     if label == 1:
#         label_str = "Competent"
#     else:
#         label_str = "Non-Competent"
    
#     # Compute prediction error
#     pred_probabilities = model(image)
 
#     heatmap = get_gradcam(model, image, pred_probabilities[0], IMG_SIZE)
#     plt.figure()
#     plot_heatmap(np.array(denorm_image.cpu()), pred_probabilities, heatmap, label_str)
  
#     pred = (torch.sigmoid(pred_probabilities)>0.5).int()
#     preds.extend(pred.cpu().numpy())
#     probabilities.extend(torch.sigmoid(pred_probabilities[0]).cpu().detach().numpy())
#     labels.append(label)
    

# "Metrics results"

# #COnfusion Matrix
# print(f'\nF1- score: {fbeta_score(labels, preds, beta = 1):.3f}')
# confusion_matrix = metrics.confusion_matrix(labels, preds)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
# cm_display.plot()
# plt.show()

# #ROC curve
# score = roc_auc_score(labels, probabilities)
# print(f"ROC AUC: {score:.4f}")
# plot_sklearn_roc_curve(labels, probabilities)

# #More metrics
# print(classification_report(labels, preds, target_names=['class 0', 'class 1']))
# a = classification_report(labels, preds, target_names=['class 0', 'class 1'],output_dict = True)
# TN = confusion_matrix[0, 0]
# FP = confusion_matrix[0, 1]
# FN = confusion_matrix[1, 0]
# TP = confusion_matrix[1, 1]

# print(' NPV: ' , TN/(TN + FN))
# print('specificity: ', TN/(TN+FP))
# print('accuracy:', accuracy_score(labels, preds))
 