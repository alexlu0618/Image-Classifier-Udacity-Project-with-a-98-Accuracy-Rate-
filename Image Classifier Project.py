#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.
# 
# Please make sure if you are running this notebook in the workspace that you have chosen GPU rather than CPU mode.

# In[1]:


# Import important packages 
# matplotlib inline: the output of plotting commands is displayed inline within frontends like the
# Jupyter notebook, directly below the code cell that produced it. 
#The resulting plots will then also be stored in the notebook document.
get_ipython().run_line_magic('matplotlib', 'inline')
# To use retina display mode
get_ipython().run_line_magic('config', "InlineBankend.figure_format = 'retina'")
# for plotting
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt 
# import PyTorch package, and some sub methods and fucntions, e.g. nn for building network sturcture, 
# optim for updating network weights
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
# import transform fucntion and models from torchvision. models has many pretrained models like VGG16
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.image as mpimg
from matplotlib.ticker import FormatStrFormatter
import json
import random 
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets

data_transforms = {

    'train':transforms.Compose ([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485,0.456,0.406],
                                                             std=[0.229,0.224,0.225])
                                       ]),

    'valid':transforms.Compose ([transforms.Resize(256),
                                        transforms.CenterCrop (224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485,0.456,0.406],
                                                             std=[0.229,0.224,0.225])
                                       
                                       ]),

    'test':transforms.Compose ([transforms.Resize(256),
                                      transforms.CenterCrop (224),
                                      transforms.ToTensor(),
                                       transforms.Normalize (mean=[0.485,0.456,0.406],
                                                            std=[0.229,0.224,0.225]) 
                                      ])

}
# TODO: Load the datasets with ImageFolder
image_datasets = {
'train': datasets.ImageFolder (train_dir, transform = data_transforms['train']),
'valid' : datasets.ImageFolder (valid_dir, transform = data_transforms['valid']),
'test' : datasets.ImageFolder (test_dir, transform = data_transforms['test'])
}
    
    
# TODO: Using the image datasets and the trainforms, define the dataloaders

data_loader = { 
    'train':data.DataLoader(image_datasets['train'], batch_size=64, shuffle = True),
    'valid':data.DataLoader(image_datasets['valid'], batch_size=64, shuffle = True),
    'test':data.DataLoader(image_datasets['test'], batch_size=64, shuffle = True)
}


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[26]:


#load the jason file that has all the flower names
# later this will be used for mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# In[27]:


# to read an example
cat_to_name


# In[6]:


#to have a quick look at the labels of the data in the batch
# later after the model is trained, I will use do 
# the mapping for the forecasted outputs. 
image,label = next(iter(data_loader['train']))
print(label)
print(image)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

# In[7]:


# TODO: Build and train your network
# Import models and use the pretrained weights
model = models.densenet161(pretrained=True)


# In[8]:


print(model)


# In[8]:


# all for model VGG16

#if GPU is available, will use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# freeze parameters so we don't backprop through them 
for param in model.parameters():
    param.requires_grad = False

# set up a new classfier for the model, add dropout rate 0.2

model.classifier = nn.Sequential(nn.Linear(2208, 102),
                                nn.LogSoftmax(dim=1))  
# use negative logritem for loss calculation
criterion = nn.NLLLoss()

# to set parameters for optimizer
optimizer = optim.Adam(model.classifier.parameters())

sched = optim.lr_scheduler.StepLR(optimizer, step_size=4)

#move content to the device CPU or GPU
model.to(device);


# In[9]:


epochs = 30
steps = 0
running_loss = 0
print_every = 10

for epoch in range (epochs):
    # loop through the trainloader
    for inputs, labels in data_loader['train']:
        steps += 1
        #Move input and label rensors to the default device 
        inputs, labels = inputs.to(device) , labels.to(device) 
        #zero the optimizer
        optimizer.zero_grad()
        #forwardfeed the data
        logps = model.forward(inputs)
        #calculate loss for this batch
        loss = criterion (logps, labels)
        #caculate gradient descent
        loss.backward()
        # update wrights 
        optimizer.step()
        
        # accumulate loss for this bath, step by step
        running_loss += loss.item()
        
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            #in for testing purpose, turn the dropout function
            model.eval()
            
            # do not caculate the gradient descent
            with torch.no_grad():
                # loop through the testloader for validation loss calculation purpose
                for inputs, labels in data_loader['valid']:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward (inputs)
                    batch_loss = criterion (logps,labels)
                    validation_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim =1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean (equals.type (torch.FloatTensor)).item()

                # validation accuracy calculation
         
            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Val loss: {validation_loss/len(data_loader['valid']):.3f}.. "
                f"Val accuracy: {accuracy/len(data_loader['valid']):.3f}")
            running_loss = 0
            model.train()


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[10]:


# TODO: Do validation on the test set
test_loss = 0
accuracy = 0
model.eval()

with torch.no_grad():
    for inputs, labels in data_loader['test']:
        inputs, labels = inputs.to(device),labels.to(device)
        logps = model.forward (inputs)
        ps = torch.exp(logps)
        
        batch_loss = criterion (logps,labels)
        
        test_loss += batch_loss.item()
                    
        top_p, top_class = ps.topk(1, dim =1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean (equals.type (torch.FloatTensor)).item()

print(
        f"Test loss: {test_loss/len(data_loader['test']):.3f}.. "
        f"Test accuracy: {accuracy/len(data_loader['test']):.3f}")


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[11]:


# have a quick at the model
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())


# In[12]:


# TODO: Save the checkpoint for creating a new model
# to save the state dict keys that has all the weights for different layers
torch.save(model.state_dict(), 'checkpoint_160319_best.pth') 


# In[16]:


model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {
    'arch': 'densenet161',
    'class_to_idx': model.class_to_idx, 
    'optimizer_dict':optimizer.state_dict(),
    'state_dict': model.state_dict(),
    #'hidden_units' : [1000,500]
}

torch.save(checkpoint,'checkpoint_160319_best.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[17]:


# TODO: Write a function that loads a checkpoint and rebuilds the model

check_point = torch.load('checkpoint_160319_best.pth')

def load_checkpoint(checkpoint):
    arch = checkpoint['arch']
    no_labels = len(checkpoint['class_to_idx'])
    
    #hidden_units = checkpoint['hidden_units']
    
    
    
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
                            nn.Linear (2208,102),
                            nn.LogSoftmax(dim=1))  

    model.class_to_idx = checkpoint['class_to_idx']
    
    model.load_state_dict(checkpoint['state_dict'])
    return model


# In[18]:


model_3 = load_checkpoint(check_point)


# In[19]:


#test model_3, newly loaded model
# TODO: Do validation on the test set
test_loss = 0
accuracy = 0
model_3.eval()
model_3.cuda()
with torch.no_grad():
    for inputs, labels in data_loader['test']:
        inputs, labels = inputs.to(device),labels.to(device)
        logps = model_3.forward (inputs)
        ps = torch.exp(logps) 
        batch_loss = criterion (logps,labels)
        test_loss += batch_loss.item()              
        top_p, top_class = ps.topk(1, dim =1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean (equals.type (torch.FloatTensor)).item()
print(
        f"Test loss: {test_loss/len(data_loader['test']):.3f}.. "
        f"Test accuracy: {accuracy/len(data_loader['test']):.3f}")


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[20]:


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[21]:


def processing_image (path):
    image_loader = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])    
    im = Image.open(image_path)
    im = image_loader(im).float()
    
    return im


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[22]:


def predict (image, model):

    ima = image.unsqueeze(0)  
    ima = ima.to(device)
    model.eval()
    with torch.no_grad():   
        logps = model.forward(ima)   
        ps = torch.exp(logps)                
        probs, classes = ps.topk(5, dim =1) 
        
    return probs, classes


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[28]:


def processing_label_names (labels,idx_to_classes, cat_to_name):
    c = []
    for i in labels:  
        a= idx_to_classes[i]
        c.append(a)
        
    d = []
    for i in c:
        a = cat_to_name[str(i)]
        d.append(a)

    return d


# In[38]:


def check_sanity(image_path):

    plt.rcParams["figure.figsize"] = (10,4)
    plt.subplot(211)

    image_path = image_path
    
    image1 = processing_image (image_path)

    image2 = np.array(image1)

    axs = imshow(image2, ax = plt)
    axs.axis('off')
    axs.title(cat_to_name[str(43)])
    axs.show() 
   
    probs, labels = predict(image1, model_3)

    probs = np.array(probs)
    topk_classes = np.array(labels[0])
    
    idx_to_classes = {v:k for k,v in model_3.class_to_idx.items()}   
    
    topk_classes= processing_label_names (topk_classes, idx_to_classes,cat_to_name)

    locations = [1,2,3,4,5]
    heights = probs[0].tolist()
   
    plt.bar(locations, heights, tick_label=topk_classes);

    plt.title('Forecasted Possibilities and Flower Names')
    plt.xlabel('Flower Names')
    plt.ylabel('Possibilities')
    plt.show()


# In[39]:


image_path = "flowers/test/43/image_02329.jpg"
check_sanity(image_path )


# In[ ]:





# In[ ]:




