from UNet import *
from dataloader_4 import *
from utils import *
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import torch
import gc
import copy
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

# TODO: Some missing values are represented by '__'. You need to fill these up.
# TODO: Some missing values are represented by '__'. You need to fill these up.

train_dataset_ori = TASDataset('tas500v1.1')
train_dataset_crop = TASDataset('tas500v1.1', transform_mode=['crop'])
train_dataset_flip = TASDataset('tas500v1.1', transform_mode=['flip'])
train_dataset_rotate = TASDataset('tas500v1.1', transform_mode=['rotate'])
train_dataset = ConcatDataset([train_dataset_ori, train_dataset_crop, train_dataset_flip,train_dataset_rotate ]) 

val_dataset = TASDataset('tas500v1.1', eval=True, mode='val')
test_dataset = TASDataset('tas500v1.1', eval=True, mode='test')
plot_dataset = TASDataset('tas500v1.1', eval=True, mode='plot')

train_loader = DataLoader(dataset=train_dataset, batch_size= 4, num_workers=1, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 4, num_workers=1, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= 4, num_workers=1, shuffle=False)
plot_loader = DataLoader(dataset=plot_dataset, batch_size= 1, num_workers=1, shuffle=False)

model_name = 'U_NET_p5c_1.pt'

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   

epochs =  20   
#nSamples = [22829796, 78633099, 5283067, 1152484, 13038040, 881615, 185824, 20445, 6125114, 137236]
#normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
#normedWeights = torch.FloatTensor(normedWeights).to(device)
criterion = nn.CrossEntropyLoss()#normedWeights # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
n_class = 10
fcn_model = U_Net(n_class=n_class)
fcn_model.apply(init_weights)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # determine which device to use (gpu or cpu)
fcn_model = fcn_model.to(device) #transfer the model to the device

def train(epochs = 20, learning_rate=0.0001):
    
    optimizer = optim.Adam(fcn_model.parameters(), lr=learning_rate) # choose an optimizer
    
    print("Beginning Training!")
    print("Using device: {}".format(device))
    best_iou_score = 0.0
    total_train_loss, total_valid_loss = [], []
    for epoch in range(epochs):
        train_loss = []
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            # reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device) #transfer the input to the same device as the model's
            labels = labels.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            outputs = fcn_model(inputs) #we will not need to transfer the output, it will be automatically in the same device as the model's!
            
            loss = criterion(outputs, labels) #calculate loss
            
            # backpropagate
            loss.backward()
            # update the weights
            optimizer.step()
            train_loss.append(loss.item())

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        total_train_loss.append(np.mean(train_loss))

        val_loss, current_miou_score, val_acc = val(epoch)
        total_valid_loss.append(val_loss)
        
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            #save the best model
            torch.save(fcn_model, model_name)
    return total_train_loss, total_valid_loss
    

def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            output = fcn_model(input)

            loss = criterion(output, label) #calculate the loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 
            
            pred = torch.argmax(output, dim=1) # Make sure to include an argmax to get the prediction from the outputs of your model
            
            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
            accuracy.append(pixel_acc(pred, label)) # Complete this function in the util
  

    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(losses), np.nanmean(mean_iou_scores), np.mean(accuracy)

def test():
    #TODO: load the best model and complete the rest of the function for testing
    fcn_model = torch.load(model_name)
    fcn_model.eval()
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            output = fcn_model(input)

            loss = criterion(output, label) #calculate the loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 

            pred = torch.argmax(output, dim=1) # Make sure to include an argmax to get the prediction from the outputs of your model

            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc(pred, label)) # Complete this function in the util


    print(f"Loss is {np.mean(losses)}")
    print(f"IoU is {np.mean(mean_iou_scores)}")
    print(f"Pixel is {np.mean(accuracy)}")
    
def test_visualization():
    class2color = {}
    for k, v in test_dataset.color2class.items():
        class2color[v] = k

    fcn_model = torch.load(model_name)
    fcn_model.eval()  # Don't forget to put in eval mode !
    ts = time.time()
    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label, real_image) in enumerate(plot_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.type(torch.LongTensor).to(device) #transfer the labels to the same device as the model's

            output = fcn_model(input)
            pred = torch.argmax(output, dim=1) 
            
        
    
            imgs = []
            for rows in pred[0]:
                for col in rows:
                    col = int(col)
                    imgs.append(class2color[col])
            imgs = np.asarray(imgs).reshape(pred.shape[1], pred.shape[2], 3)
            outputimg = PIL.Image.fromarray(np.array(imgs, dtype=np.uint8))
            plt.axis('off')
            plt.imshow(real_image[0])
            plt.imshow(outputimg, alpha=0.8)
            
            plt.title('Output Image')
            plt.show()

            imgs = []
            for rows in label[0]:
                for col in rows:
                    col = int(col)
                    imgs.append(class2color[col])
            imgs = np.asarray(imgs).reshape(pred.shape[1], pred.shape[2], 3)
            outputimg = PIL.Image.fromarray(np.array(imgs, dtype=np.uint8))
            plt.axis('off')
            plt.imshow(real_image[0])
            plt.imshow(outputimg, alpha=0.8)
            
            plt.title('Label Image')
            plt.show()
    


if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
    test()
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()