import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import numpy as np
from nntraining import Net
#with torch.no_grad():

#    index = 256
#item = datasets(index)
#image = item[0]
#true_target = item[1]

args={}
kwargs={}
args['batch_size']=1000
args['test_batch_size']=1000
args['epochs']=2  #The number of Epochs is the number of times you go through the full dataset. 
args['lr']=0.5 #Learning rate is how fast it will decend. 
args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args['seed']=1 #random seed
args['log_interval']=10
args['cuda']=False



model = Net()
if args['cuda']:
    model.cuda()


img = cv2.imread('paint1.png', cv2.IMREAD_GRAYSCALE)
res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

print(res.shape)
imgtor = -torch.from_numpy(res).unsqueeze(0).unsqueeze(0).to(torch.float32)/255 + 1

print(imgtor)

transform = transforms.Normalize((0.1307,), (0.3081,))

imgtor = transform(imgtor)


model.load_state_dict(torch.load("../model.pt"))

model.eval()

res = model(imgtor)


print(res)
predicted_class = np.argmax(res.detach().numpy())
print(predicted_class)
