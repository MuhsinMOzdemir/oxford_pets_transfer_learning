# Importing the libraries
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import random_split

# Defining the device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Defining the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizes teh image to 224x224
    transforms.ToTensor(),  # Converts the image into a pytorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # Since it is pre-trained we use these specific numbers to normalize. Otherwise calculate it yourself.
])  # This is Mean               # This is Standard Deviation
train_set = datasets.OxfordIIITPet(root="data", split="trainval", download=True, transform=transform)
test_set = datasets.OxfordIIITPet(root="data", split="test", download=True, transform=transform)
#######################################################################################################################
# root is the path to the folder where the data is stored. So in this case program creates a folder called data.      #
# split is the way data will be used. Amount is decided by the owner of dataset in this case.                         #
# OxfordIIITPet splits 3,680 images to trainval and 3,669 into test.                                                  #
# Trainval will be used for learning and tuning while test is used for testing.                                       #
# The reason split is pre-determined is so that in research the results are comparable. Similar concept to same seeds.#
#######################################################################################################################

# print(len(train_set), len(test_set))  # 3,680 and 3,669 images
# print(len(train_set.classes))  # 37 different breeds of pets
#
# img, label = train_set[0]
# print(img.shape)  # 3 colors x 224 pixels x 224 pixels
# print(img.min(),
#       img.max())  # After .ToTensor() your image is between 0 and 1 so when we do (input-mean)/std we get values between -2.12 and 2.64. For input = 0 and 1 respectively.
# print(train_set.classes[label])  # the pet's breed
if __name__ == "__main__":
    print("It is working")
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)
    ####################################################################################################################################################################
    # batch_size is how many images at a time model trains/test on.                                                                                                    #
    # shuffle basically mixes the images before feeding it to cpu. if it is false first x amount images would be the same breed.                                       #
    # Which would simply the model's complexity a lot to a point where it is just predicting what a breed is according to what the last image was.                     #
    ####################################################################################################################################################################
    # num_workers prepares each batch. Each batch has to be opened, decoded, go through transform() and get stacked(32 images).
    # if num_workers=0, then first a batch gets prepared and then model gets trained. if num_workers=1, one batch is trained while model is trained.
    # if  num_workers=4, four batches are being prepared while the model is being prepared.
    # When a batch is complete the size becomes [32, 3, 224, 224] since 32 images are stacked into one tensor.
    # if torch.device is dedicated to gpu then num_workers get handled by cpu while training model by gpu.
    # After cpu finishes a batch it sends to gpu, which has its own cost. That's why when batch is small enough training the model on gpu makes less sense.
    # images.to(device) ===> copying to GPU
    ####################################################################################################################################################################

    # The model with pretrained weights
    net = resnet18(weights=ResNet18_Weights.DEFAULT)

    # print(net) # This would print the whole model.
    # print(net.fc) # This would jsut print the last layer
    # Which prints Linear(in_features=512, out_features=1000, bias=True)
    # Everything before this layer has boiled the image down to 512 numbers
    # 1000 out — one score per ImageNet class. Golden retriever, tabby cat, school bus, and 997 others.
    # we need to change the net.fc out_features to 37 since we don't need 1000 different objects.

    for param in net.parameters():  # Freezes the weights for each layer
        param.requires_grad = False

    net.fc = nn.Linear(512, 37)  # Change is done here so that the last layer weights are not frozen
    # There should be (512 x 37) +37 parameters in the model that are active. In read me explain why we did this and delete this sentence.
    # The reasoning to freeze every weight but the last layer's is because we only have small amount of photos(3,680) compared to the amount of photos it took to train the model(1.2 million)
    # If we don't freeze most weights, since model has around 11.1 million parameters, the model would overfit our training data.
    # So we only let some weights to be unfrozen which is around 0.2% of 11.1 million.
    net = net.to(device)
    # Training the data model
    criterion = nn.CrossEntropyLoss()  # This is the loss function. It turns the 37 raw scores model produced into probabilities and asks how much probability is put on the right breed.
    optimizer = torch.optim.SGD(net.fc.parameters(), lr=0.01,
                                momentum=0.9)  # In this line we are setting the learning rate and momentum. The .fc here is the last layer. We could also remove it since other layers' parameters are frozen
    # This loss function punishes the wrongness harshly since it is a -log(x).
    # SGD = Stochastic Gradient Descent
    # For further research try Adam instead of SGD

    train_subset, val_subset = random_split(
        train_set, [3300, 380],
        generator=torch.Generator().manual_seed(42)
    )# Splits the training data into 3300 and 380. 3300 is the amount of training data and 380 is the amount of validation data.


    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

    # Usual epochs
    for epoch in range(5):
        net.train()  # Turn the model into training mode
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        net.eval() # Turn the model into evaluation mode
        correct = total = 0
        class_correct = [0] * 37 # Creating a class with 37 0s to count the correct amount for each breed
        class_total = [0] * 37 # Creating a class with 37 0s to count how many times each breed was shown
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                predicted = net(images).argmax(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                for label, pred in zip(labels, predicted): # Counting each breed amount
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
        print(epoch, 100 * correct / total)
        if epoch == 4:
            for i in range(37):
                if class_total[i] > 0:
                    print(train_set.classes[i], round(100 * class_correct[i] / class_total[i], 1))