
#MobileNetV3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
import numpy as np
from tqdm import tqdm
#import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.tensorboard import SummaryWriter  
import pdb

import torchvision


from dataset import read_data
import pandas as pd
import time
import datetime

from  torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_loop(dataloader, model, loss_fn, optimizer, scheduler, device):
    model.train()
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    train_loss = 0.0
    train_acc = 0.0
    for batch, sample in enumerate(pbar):
        x,labels  = sample
        x,labels  = x.to(device), labels.to(device)

        outputs = model(x)
        loss = loss_fn(outputs, labels)
        _,pred = torch.max(outputs,1)
        num_correct = (pred == labels).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #scheduler.step()

        loss = loss.item()
        acc = num_correct.item()/len(labels)
        count += len(labels)
        train_loss += loss*len(labels)
        train_acc += num_correct.item()
        pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")
        
    return train_loss/count, train_acc/count
        
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)

    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    test_loss = 0.0
    test_acc = 0.0
    gt = []
    pd = []
    fpr = dict()
    tpr = dict()
    roc_auc = []
    n_classes = 6          #no. of classes
    with torch.no_grad():
        for batch, sample in enumerate(pbar):
            x,labels = sample
            x,labels = x.to(device), labels.to(device)

            outputs = model(x)
            loss = loss_fn(outputs, labels)
            _,pred = torch.max(outputs,1)
            num_correct = (pred == labels).sum()
            loss = loss.item()
            acc = num_correct.item()/len(labels)
            count += len(labels)
            test_loss += loss*len(labels)
            test_acc += num_correct.item()
            gt.extend(labels.cpu().numpy())
            pd.extend(outputs.cpu().numpy())

            pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")

    gt = np.array(gt); pd = np.array(pd)
    gt = label_binarize(np.array(gt), classes=[0, 1, 2, 3, 4, 5])        # class number list
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(gt[:, i], pd[:, i])
        roc_auc.append(auc(fpr[i], tpr[i]))
    aucavg = np.mean(roc_auc)
    print("AUC: {}".format(roc_auc))

    return test_loss/count, test_acc/count, aucavg


   
# Function to initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
            
# Function to reset the classifier
def reset_classifier(model, num_classes):


    model.classifier[-2] = nn.Dropout(p=0.3, inplace=True)

    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes, bias=True) if num_classes > 0 else nn.Identity()
    
    model.classifier[-1].apply(init_weights)
    



if __name__ == '__main__':

    device =  torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print('Device:', device)
    save_dir = './HD_384_B_6'
    data_dir = '/DATA/ishwar_2221cs30/Work_2/Apple2021/data'
    writer = SummaryWriter(save_dir)
    batchsize = 64
    n_epochs = 200
    Lr = 3e-4    
    
    evaluate_train = False
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    train_generator,test_generator = read_data(batchsize, data_dir, num_workers=8)
    
    
    total_cnt = len(train_generator.dataset)
    num_steps_per_epoch = total_cnt // batchsize
    steps_per_epoch=len(train_generator)
    #print("num_steps_per_epoch : ", num_steps_per_epoch)
    print("steps :", steps_per_epoch)
    
    # Load the pre-trained Swin Transformer model
    model = mobilenet_v3_large(weights= MobileNet_V3_Large_Weights.DEFAULT, progress=True) 
     
    # Reset the classifier for the new task with, for example, 10 classes
    reset_classifier(model, num_classes=6) 
    
    model = model.to(device)
    
    # Verify the model structure
    print(model)
    
    print("There are", sum(p.numel() for p in model.parameters()), "parameters.")
    print("There are", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")
    if torch.cuda.is_available() and torch.cuda.device_count()>1:
      print("Using {} GPUs.".format(torch.cuda.device_count()))
      model = torch.nn.DataParallel(model)
    
    
    criterion = nn.CrossEntropyLoss().to(device)
    weight_p, bias_p = [],[]
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p +=[p]
        else:
            weight_p +=[p]
            
            

    optimizer = torch.optim.AdamW(model.parameters(),weight_decay=0.001, lr=Lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True)    
    
 

    idx_best_loss = 0
    idx_best_acc = 0
    idx_best_auc = 0
    
    log_train_loss = []
    log_train_acc = []
    log_val_loss = []
    log_val_acc = []
    log_auc = []
    
    print("Start training for 200 epochs")

    start_time = time.time()    
    
    for epoch in range(1, n_epochs+1):
    
        epoch_start_time = time.time()  # Start timer
            
        # print("===> Epoch {}/{}, learning rate: {}".format(epoch, n_epochs, scheduler.get_last_lr()))
        print("===> Epoch {}/{}, learning rate: {}".format(epoch, n_epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = train_loop(train_generator, model, criterion, optimizer, scheduler, device)
        if evaluate_train:
            train_loss, train_acc, test_auc = test_loop(train_generator, model, criterion, device)
        val_loss, val_acc, val_auc = test_loop(test_generator, model, criterion, device)
        
        epoch_end_time = time.time()  # End timer
        
        if(epoch==1):
            epoch_time = epoch_end_time - epoch_start_time  # Calculate epoch time
            epoch_time_str=str(datetime.timedelta(seconds=int(epoch_time)))
            print("=============================================================")
            print('one epoch time {}'.format(epoch_time_str))        
        
                
        
        print("Training loss: {:f}, acc: {:f}".format(train_loss, train_acc))
        print("Val loss: {:f}, acc: {:f}".format(val_loss, val_acc))
        print("Val AUC: {:.2f}".format(val_auc))
        writer.add_scalar("Trainloss", train_loss, epoch)
        writer.add_scalar("Valloss", val_loss, epoch)
        writer.add_scalar("Trainacc", train_acc, epoch)
        writer.add_scalar("Valacc", val_acc, epoch)
        writer.add_scalar("ValAUC", val_auc, epoch)


        scheduler.step(val_loss)
                
        
        log_train_loss.append(train_loss)
        log_train_acc.append(train_acc)
        log_val_loss.append(val_loss)
        log_val_acc.append(val_acc)
        log_auc.append(val_auc)
        
        if val_loss <= log_val_loss[idx_best_loss]:
            print("Save loss-best model.")
            torch.save(model.state_dict(), os.path.join(save_dir, 'loss_model.pth'))
            idx_best_loss = epoch - 1
        
        if val_acc >= log_val_acc[idx_best_acc]:
            print("Save acc-best model.")
            torch.save(model.state_dict(), os.path.join(save_dir, 'acc_model.pth'))
            idx_best_acc = epoch - 1
        
        if val_auc >= log_auc[idx_best_auc]:
            print("Save auc-best model.")
            torch.save(model.state_dict(), os.path.join(save_dir, 'auc_model.pth'))
            idx_best_auc = epoch - 1

        print("")
        
    print("=============================================================")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))        
        

    print("=============================================================")

    print("Loss-best model training loss: {:f}, acc: {:f}".format(log_train_loss[idx_best_loss], log_train_acc[idx_best_loss]))   
    print("Loss-best model val loss: {:f}, acc: {:f}".format(log_val_loss[idx_best_loss], log_val_acc[idx_best_loss]))                
    print("Acc-best model training loss: {:4f}, acc: {:f}".format(log_train_loss[idx_best_acc], log_train_acc[idx_best_acc]))  
    print("Acc-best model val loss: {:f}, acc: {:f}".format(log_val_loss[idx_best_acc], log_val_acc[idx_best_acc]))              
    print("Final model training loss: {:f}, acc: {:f}".format(log_train_loss[-1], log_train_acc[-1]))                 
    print("Final model val loss: {:f}, acc: {:f}".format(log_val_loss[-1], log_val_acc[-1]))           
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    
    
    print("=============================================================")
    
    print("Best loss achieved on epoch:", idx_best_loss+1) # conside counting of epcoch as 1,2,3 ...
    print("Best Acc achieved on epoch:", idx_best_acc+1)     
    
    
    
    
    log_train_loss = np.array(log_train_loss)
    log_train_acc = np.array(log_train_acc)
    log_val_loss = np.array(log_val_loss)
    log_val_acc = np.array(log_val_acc)
    log_auc = np.array(log_auc)


    # Create a dictionary to hold the data
    data = {
        'train_loss': log_train_loss,
        'train_acc': log_train_acc,
        'val_loss': log_val_loss,
        'val_acc': log_val_acc,
        'val_auc': log_auc
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv('train_log.csv', index=False)


    
    plt.figure(figsize=(10, 4))     # 10,4
    plt.subplot(131)
    plt.plot(np.arange(1, n_epochs + 1), log_train_loss)  # train loss (on epoch end)
    plt.plot(np.arange(1, n_epochs + 1), log_val_loss)         #  test loss (on epoch end)
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.xlim([0, n_epochs])
    plt.legend(['Train', 'val'], loc="upper right")
    
    plt.subplot(132)
    plt.plot(np.arange(1, n_epochs + 1), log_train_acc)  # train accuracy (on epoch end)
    plt.plot(np.arange(1, n_epochs + 1), log_val_acc)         #  test accuracy (on epoch end)
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.xlim([0, n_epochs])
    plt.legend(['Train', 'val'], loc="lower right")    

    plt.subplot(133)
    plt.plot(np.arange(1, n_epochs + 1), log_auc)         #  test AUC (on epoch end)
    plt.title("AUC")
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid()
    plt.xlim([0, n_epochs])
    
    # Adjust the spacing
    plt.subplots_adjust(wspace=0.4)  # Adjust the width spacing between subplots

    
    plt.savefig(os.path.join(save_dir, 'plot.png'))
    plt.show()
    

    


