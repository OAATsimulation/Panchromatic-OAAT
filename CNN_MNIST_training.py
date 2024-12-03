#LeNet
import torch
from torch.utils import data
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor

#加载数据
batch_size = 256
def load_data_fashion_mnist(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../Datasets',train = True,transform = trans)
    mnist_test = torchvision.datasets.FashionMNIST(root='../Datasets',train = False,transform = trans)

    return (data.DataLoader(mnist_train,batch_size,shuffle = True), data.DataLoader(mnist_test,10000,shuffle = False))

train_iter, test_iter = load_data_fashion_mnist(batch_size)

#建立网络
class Reshape(torch.nn.Module):
    def forward(self,x):
        return x.reshape((-1,1,28,28))

net = nn.Sequential(Reshape(),
                    nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size = 2, stride=2),
                    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2,stride=2),
                    nn.Flatten(),
                    nn.Linear(16*5*5,120),nn.Sigmoid(),
                    nn.Linear(120,84),nn.Sigmoid(),
                    nn.Linear(84,10))

#初始化参数
def num_right_train_pred(X,y):
    y_hat = net(X).argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)

#损失函数
lr = 0.1
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr)
num_epochs = 20
for epoch in range(num_epochs):
    train_l,train_acc,y_numel = 0.0,0.0,0
    for X,y in train_iter:
        optimizer.zero_grad()
        l = loss(net(X),y)
        l.sum().backward()
        optimizer.step()
        train_l += l.sum()
        train_acc += num_right_train_pred(X,y)
        y_numel += y.numel()
    print(('epoch: ',epoch+1,'\t','train_l: ',train_l/y_numel,'\t','train_acc: ',train_acc/y_numel))