import numpy as np
import torch

###########################################################################
    # Implement the train/test module.
    # Understand train/test codes, and fill in the blanks.
def train_model(trainloader, model, criterion, optimizer, scheduler, device):
    model.train()
    for i, (inputs, labels) in enumerate(trainloader):
        from datetime import datetime

        inputs = inputs.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        criterion = criterion.cuda()
        ##########################################
        ############# fill in here -> train
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        ####### Hint :
        ####### 1. Get the output out of model, and Get the Loss
        ####### 3. optimizer
        ####### 4. backpropagation
        #########################################

def accuracy_check(label, pred):
    ims = [label, pred]
    np_ims = []
    for item in ims:
        item = np.array(item)
        np_ims.append(item)
    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)
    return accuracy / len(np_ims[0].flatten())

def accuracy_check_for_batch(labels, preds, batch_size):
    total_acc = 0
    for i in range(batch_size):
        total_acc += accuracy_check(labels[i], preds[i])
    return total_acc/batch_size

def get_loss_train(model, trainloader, criterion, device):

    model.eval()
    total_acc = 0
    total_loss = 0
    for batch, (inputs, labels) in enumerate(trainloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device = device, dtype = torch.int64)
            inputs = inputs.float()
            ##########################################
            ############# fill in here -> (same as validation, just printing loss)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            ####### Hint :
            ####### Get the output out of model, and Get the Loss
            ####### Think what's different from the above
            #########################################
            outputs = np.transpose(outputs.cpu(), (0,2,3,1))
            preds = torch.argmax(outputs, dim=3).float()
            acc = accuracy_check_for_batch(labels.cpu(), preds.cpu(), inputs.size()[0])
            total_acc += acc
            total_loss += loss.cpu().item()
    return total_acc/(batch+1), total_loss/(batch+1)

from PIL import Image
def val_model(model, valloader, criterion, device, dir):

    cls_invert = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0),  # 0:background, 1:aeroplane, 2:bicycle
                  3: (128, 128, 0), 4: (0, 0, 128), 5: (128, 0, 128),  # 3:bird, 4:boat, 5:bottle
                  6: (0, 128, 128), 7: (128, 128, 128), 8: (64, 0, 0),  # 6:bus, 7:car, 8:cat
                  9: (192, 0, 0), 10: (64, 128, 0), 11: (192, 128, 0),  # 9:chair, 10:cow, 11:diningtable
                  12: (64, 0, 128), 13: (192, 0, 128), 14: (64, 128, 128),  # 12:dog, 13:horse, 14:motorbike
                  15: (192, 128, 128), 16: (0, 64, 0), 17: (128, 64, 0),  # 15:person, 16:pottedplant, 17:sheep
                  18: (0, 192, 0), 19: (128, 192, 0), 20: (0, 64, 128),  # 18:sofa, 19:train, 20:tvmonitor
                  21: (224, 224, 192)}
    total_val_loss = 0
    total_val_acc = 0
    n=0

    for batch, (inputs, labels) in enumerate(valloader):
        with torch.no_grad():

            inputs = inputs.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            ##########################################
            ############# fill in here -> (validation)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            ####### Hint :
            ####### Get the output out of model, and Get the Loss
            ####### Think what's different from the above
            #########################################

            outputs = np.transpose(outputs.cpu(), (0, 2, 3, 1))
            preds = torch.argmax(outputs, dim=3).float()

            acc = accuracy_check_for_batch(labels.cpu(), preds.cpu(), inputs.size()[0])
            total_val_acc += acc
            total_val_loss += loss.cpu().item()

            for i in range(preds.shape[0]):
                temp = preds[i].cpu().data.numpy()
                temp_l = labels[i].cpu().data.numpy()
                temp_rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
                temp_label = np.zeros((temp.shape[0], temp.shape[1], 3))

                for j in range(temp.shape[0]):
                    for k in range(temp.shape[1]):
                        ##########################################
                        ############# fill in here 
                        temp_rgb[j][k] = cls_invert[temp[j][k]]
                        temp_label[j][k] = cls_invert[temp_l[j][k]]
                        ####### Hint :
                        ####### convert segmentation mask into r,g,b (both for image and predicted result)
                        ####### image should become temp_rgb, result should become temp_label
                        ####### You should use cls_invert[]
                        #########################################

                img = inputs[i].cpu()
                img = np.transpose(img, (2, 1, 0))

                img_print = Image.fromarray(np.uint8(temp_label))
                mask_print = Image.fromarray(np.uint8(temp_rgb))

                img_print.save(dir + str(n) + 'label' + '.png')
                mask_print.save(dir + str(n) + 'result' + '.png')

                n += 1

    return total_val_acc/(batch+1), total_val_loss/(batch+1)