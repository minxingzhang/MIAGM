import argparse
import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torchvision import  models
import torchvision.utils as tvu
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from torchmetrics import ConfusionMatrix

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def main(args):
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # Decide GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.set_num_threads(1)
    sys.path.append(os.getcwd())
    torch.set_printoptions(sci_mode=False)

    confmat = ConfusionMatrix(num_classes=2).to(device)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
        
    # Training
    print("train starts")
    model.train()

    train_in_set = torch.load(args.train_in_path)
    train_out_set = torch.load(args.train_out_path)
    train_loader = torch.utils.data.DataLoader(train_in_set+train_out_set, batch_size=args.batchsize, shuffle=True)

    test_in_set = torch.load(args.test_in_path)
    test_out_set = torch.load(args.test_out_path)
    test_loader = torch.utils.data.DataLoader(test_in_set+test_out_set, batch_size=args.batchsize, shuffle=False)

    for epoch in range(args.epochs):
        loss = 0
        for label, data in train_loader:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
                    
            outputs = model(data)
                    
            train_loss = criterion(outputs, label)
                    
            train_loss.backward()
                    
            optimizer.step()
                    
            loss += train_loss.item()
            
        loss = loss / len(train_loader)
        print("Epoch : {}/{}, loss = {:.6f}".format(epoch+1, args.epochs, loss))
            
        # Evaluation
        print("test starts")
        model.eval()

        correct = 0
        total = 0
        
        preds = torch.Tensor(0).to(device)
        targets = torch.Tensor(0).to(device)

        with torch.no_grad():
            for label, data in test_loader:
                data = data.to(device)
                label = label.to(device)
                
                outputs = model(data)
                
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                preds = torch.cat((preds, torch.squeeze(predicted)))
                targets = torch.cat((targets, torch.squeeze(label)))
                
        print("Epoch : {}/{}, accuracy of the network on original dataset: {:.1f} %".format(epoch+1, args.epochs, 100*correct/total))
        targets = torch.tensor(targets).int().to(device)
        preds = preds.int().to(device)

        confusion_matrix = confmat(preds, targets)
        print("Confusion matrix:")
        print(confusion_matrix)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Black-box membership inference attack.")
    parser.add_argument("--gpu", default=0, type=int, help="Decide to run the program on which gpu")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs")
    parser.add_argument("--batchsize", default=256, type=int, help="The value of batch size")
    parser.add_argument("--lr", default=1e-5, type=float, help="The value of learning rate")

    parser.add_argument("--train-in-path", default="data/train_in.pt", type=str, help="The path of the training positives")
    parser.add_argument("--train-out-path", default="data/train_out.pt", type=str, help="The path of the training negatives")
    parser.add_argument("--test-in-path", default="data/test_in.pt", type=str, help="The path of the testing positives")
    parser.add_argument("--test-out-path", default="data/test_out.pt", type=str, help="The path of the testing negatives")
    args = parser.parse_args()
    
    main(args)
