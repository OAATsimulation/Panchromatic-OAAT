import numpy as np
import torch
from torch.utils import data
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchsummary import summary
import matplotlib.pyplot as plt
import time
import pandas as pd
import colour
import cv2 as cv
import os

Train_model = 0
#加载数据
batch_size = 128
num_batches = 50000/batch_size
device = 'cuda'

#设置随机种子，保证结果可重复
import random
random.seed(20)
np.random.seed(15)
torch.manual_seed(20)
torch.cuda.manual_seed(20)

#自写transform方法，用于修改测试集图片
class image_add(object):            #图像加法
    def __init__(self,rgb):
        self.rgb = np.array(rgb)/255
        self.zeros_array = np.zeros((32,16,3))
        self.stimuli_array = np.append(self.zeros_array + np.array(rgb),self.zeros_array,axis = 1)
    def __call__(self,image_tensor):
        image_array = np.transpose(image_tensor.numpy(), (1, 2, 0))
        cond_array = image_array + self.stimuli_array
        index_tuple = np.where(cond_array > 1)
        cond_array[index_tuple] = 1
        image_tensor = torch.tensor(np.transpose(cond_array,(2,0,1)))
        return image_tensor

class image_multiply(object):       #正片叠底模式
    def __init__(self,rgb,white):
        self.rgb = np.array(rgb)
        self.stimuli_array_blue = np.zeros((32,16,3)) + np.array(rgb)/255
        self.stimuli_array_white = np.zeros((32,16,3)) + np.array(white)/255
        self.stimuli_array = np.append(self.stimuli_array_blue, self.stimuli_array_white, axis = 1)
    def __call__(self,image_tensor):
        image_array = np.transpose(image_tensor.numpy(), (1, 2, 0))
        cond_array = image_array * self.stimuli_array
        image_tensor = torch.tensor(np.transpose(cond_array,(2,0,1)))
        return image_tensor

class image_softlight(object):          #柔光混合模式
    def __init__(self,rgb,white):
        if type(rgb).__name__ == type(white).__name__ == 'list':
            self.rgb = np.array(rgb)
            self.stimuli_array_blue = np.zeros((32,16,3)) + np.array(rgb)
            self.stimuli_array_white = np.zeros((32,16,3)) + np.array(white)
            self.stimuli_array = np.append(self.stimuli_array_blue, self.stimuli_array_white, axis = 1)
        if type(rgb).__name__ == type(white).__name__ == 'ndarray':
            self.rgb = rgb
            self.white = white
            self.stimuli_array = np.append(self.rgb, self.white, axis=1)
    def __call__(self,image_tensor):
        image_array = np.trunc(np.transpose(image_tensor.numpy(), (1, 2, 0)) * 255)
        mask1 = self.stimuli_array > 128
        mask2 = self.stimuli_array <= 128
        image_array[mask1] = image_array[mask1] + (2 * self.stimuli_array[mask1] - 255) * (
                    np.sqrt(image_array[mask1] / 255) * 255 - image_array[mask1]) / 255
        image_array[mask2] = image_array[mask2] * self.stimuli_array[mask2]/128 + (255-2 * self.stimuli_array[mask2]) \
                             * np.square(image_array[mask2]/255)
        image_tensor = torch.tensor(np.transpose(image_array,(2,0,1))/255)
        return image_tensor

class print_image(object):
    def __init__(self):
        self.a = 1
    def __call__(self,image_tensor):
        show = ToPILImage()
        show(image_tensor).show()
        return image_tensor

#蓝光、白光数据结构处理
def data_process(on,index):
    white_file_RGB = pd.read_excel('{}_white_data_node_RGB{}.xlsx'.format(on,index), sheet_name=[0], header=None, index_col=None)
    blue_file_RGB = pd.read_excel('{}_blue_data_node_RGB{}.xlsx'.format(on,index), sheet_name=[0], header=None, index_col=None)

    whiteRGBdata = white_file_RGB[0].to_numpy()
    blueRGBdata = blue_file_RGB[0].to_numpy()

    # #暂时扩充数据
    # blue_add = blueRGBdata[0,0:9].copy().reshape((1,9))
    # white_add = whiteRGBdata[0,0:9].copy().reshape((1,9))
    # blueRGBdata = np.append(blueRGBdata,blue_add,axis=1)
    # whiteRGBdata = np.append(whiteRGBdata,white_add,axis=1)

    blueRGBdata = blueRGBdata.reshape((128, 3))
    np.random.shuffle(blueRGBdata)
    blueRGBdata = np.tile(blueRGBdata,4)
    blueRGBdata = blueRGBdata.reshape((32, 16, 3))

    whiteRGBdata = whiteRGBdata.reshape((128, 3))
    np.random.shuffle(whiteRGBdata)
    whiteRGBdata = np.tile(whiteRGBdata, 4)
    whiteRGBdata = whiteRGBdata.reshape((32, 16, 3))
    return blueRGBdata,whiteRGBdata
blueRGBdata,whiteRGBdata = data_process(on='o',index=4)
#----------------------------------------------------------------------------------------------------------------------
#定义数据增强，导入数据
train_trans = [transforms.ColorJitter(brightness=0.4,contrast = 0.4),
               transforms.Resize(40),
               transforms.RandomResizedCrop(32,scale=(0.5625,1),ratio=(1.0,1.0)),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])]    #对训练集图像进行增强
test_trans = [transforms.ToTensor()]
              # image_softlight(blueRGBdata,whiteRGBdata),
              # transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])]
train_trans = transforms.Compose(train_trans)
test_trans = transforms.Compose(test_trans)
CIFAR_train = torchvision.datasets.CIFAR10(root='../Datasets',train = True,transform = train_trans,download=True)
CIFAR_test = torchvision.datasets.CIFAR10(root='../Datasets',train = False,transform = test_trans,download=True)
train_iter = data.DataLoader(CIFAR_train,batch_size,shuffle = True,drop_last=False)
test_iter = data.DataLoader(CIFAR_test,batch_size,shuffle = False,drop_last=False)


#定义残差块
class ResNet(nn.Module):
    def __init__(self,input_channel,output_channel,use_1conv = False,stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1,stride=stride)
        self.conv2 = nn.Conv2d(output_channel,output_channel,kernel_size=3,padding=1)
        if use_1conv:
            self.conv3 = nn.Conv2d(input_channel,output_channel,kernel_size=1,stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)
def resnet_block(input_channel,output_channel,num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(ResNet(input_channel,output_channel,use_1conv=True,stride=2))        #一般设置为通道数翻倍，图片大小减半
        else:
            blk.append(ResNet(output_channel,output_channel))
    return blk

#设置网络结构
b1 = nn.Sequential(nn.Conv2d(3,64,kernel_size=3,padding=1,stride=1,bias=False),
                   nn.BatchNorm2d(64),nn.ReLU())                                    #删除第一层卷积后的最大池化层
b2 = nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3 = nn.Sequential(*resnet_block(64,128,2))
b4 = nn.Sequential(*resnet_block(128,256,2))
b5 = nn.Sequential(*resnet_block(256,512,2))
net = nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Dropout(p=0.5),nn.Linear(512,10))
net = net.to(device)
# summary(net,(3,32,32))
# net = torch.load('Model/ResNet_CIFAR_03.pth')

#初始化参数
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        # nn.init.normal_(m.bias,std=0.01)
        # nn.init.normal_(m.weight,std=0.1)
net.apply(init_weights)

#损失函数
lr = 0.1
lr_period = 5
lr_decay = 0.5
wd = 5e-4
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=wd)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period,lr_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=150,last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95,last_epoch=-1,verbose=True)
num_epochs = 150

#准确率计算函数
def num_right_train_pred(X,y):
    y_hat = net(X).argmax(axis=1)
    y_hat2 = y_hat.cpu()
    # np.savetxt(r'C:\Users\HAHA\Desktop\predicted_label_indexs.csv', y_hat2, delimiter=',')            # 这里拷贝一份预测值并输出，已知标签值时，通过比较可以得到错误预判的图片的索引，便于图片输出
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def test_model(net,test_iter,device,output_matrix = True):
    net.eval()
    with torch.no_grad():
        confusion_matrix = np.zeros((10, 10))
        test_acc, y_numel = 0.0, 0
        for X, y in test_iter:
            X = X.type(torch.cuda.FloatTensor)
            y = y.to(device)
            test_acc += num_right_train_pred(X, y)
            y_numel += float(y.numel())

            #返回混淆矩阵
            if output_matrix:
                y_hat = net(X).argmax(axis=1)
                for i,j in zip(y_hat,y):
                    confusion_matrix[j][i] += 1
        print('test_acc: ', test_acc / y_numel)
        if output_matrix:
            return confusion_matrix
        else:
            return None

def train_model(net,num_epochs,train_iter,test_iter,loss,optimizer,scheduler,device,num_batches):
    starttime = time.time()
    for epoch in range(num_epochs):
        net.train()
        epochtime_start = time.time()
        train_l, train_acc, y_numel = 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            optimizer.step()
            train_l += float(l.sum())
            train_acc += num_right_train_pred(X, y)
            y_numel += float(y.numel())
        train_acc = train_acc / y_numel
        train_l = train_l / num_batches
        scheduler.step()
        epochtime_end = time.time()
        print('epoch: ', epoch + 1, '\t', 'train_l: {0:.6f}'.format(train_l), '\t',
              'train_acc: {0:.4f}'.format(train_acc), '\t',
              'time: {0:.1f} min'.format((epochtime_end - epochtime_start) / 60))
        if (epoch + 1) % 3 == 0:
            test_model(net,test_iter,device)
            torch.save(net, 'Model/ResNet_CIFAR_05.pth')
        if (epoch + 1) % 30 == 0:                                   #每20个周期重载一次数据（试试有没有重新增强的效果）
            CIFAR_train = torchvision.datasets.CIFAR10(root='../Datasets', train=True, transform=train_trans,
                                                       download=True)
            train_iter = data.DataLoader(CIFAR_train, batch_size, shuffle=True, drop_last=False)
    endtime = time.time()
    print('train_time: {0:.1f}'.format((endtime - starttime) / 60))

#----------------------------------------------------------------------------------------------------------------------

def draw_confusion_matrix(confusion_matrix,output_matrix_path):
    #输出混淆矩阵
    if type(confusion_matrix).__name__ != 'NoneType':
        plt.imshow(confusion_matrix,cmap = 'Blues')
        # for i in range(0,10):
        #     for j in range(0,10):
        #         plt.text(i-0.4,j,s = str(int(confusion_matrix[j][i])))
        plt.xticks(np.arange(0,10,1),color = 'black',labels=['airplane','automobile','bird','cat','deer','dog','fog','horse','ship','truck'],
                   rotation = 45,fontproperties = 'Arial',size = 12)
        plt.yticks(np.arange(0, 10, 1),color = 'black',labels=['airplane','automobile','bird','cat','deer','dog','fog','horse','ship','truck'],
                   rotation = 45,fontproperties = 'Arial',size = 12)
        # plt.xlabel('Predict',fontproperties = 'Arial',size = 25)
        # plt.ylabel('True',fontproperties = 'Arial',size = 25)
        plt.colorbar()
        plt.clim(0,1000)
        plt.savefig(output_matrix_path,dpi = 300)
        plt.pause(0)
        print(confusion_matrix)
    else:
        return None

def pick_test_image(test_iter,image_label):
    #输出测试集特定类别的图片，每一类别的图片刚好一千张
    #为保证图片正常显示，要取消对测试集图片张量的normalize处理
    image_list = []
    for X, y in test_iter:
        X = np.transpose(X.numpy(),(0,2,3,1))
        y = y.numpy()
        image_index = np.where(y == image_label)
        for i in image_index[0]:
            image_list.append(X[i])
    image_array = np.array(image_list)
    return image_array

def test_picked_image(image_array,image_label):
    #测试网络对提取出图片的分类准确率
    image_tensor = torch.tensor(np.transpose(image_array,(0,3,1,2)))
    net.eval()
    with torch.no_grad():
        test_acc, y_numel = 0.0, 0
        y = torch.tensor(np.zeros(1000)+image_label)
        y = y.to(device)
        X = image_tensor
        X = X.type(torch.cuda.FloatTensor)
        test_acc = num_right_train_pred(X,y)/1000
        print('test_acc_picked:',test_acc)

def output_picked_image(image_array,interval,path,image_index,index_array):
    if interval:
        for i in range(0,1000,interval):
            plt.imshow(image_array[i])
            plt.axis('off')
            plt.savefig('{}/{}.png'.format(path,i), bbox_inches='tight',pad_inches = 0)
            plt.close()
    elif image_index:
        plt.imshow(image_array[image_index])
        plt.axis('off')
        plt.savefig('{}/{}.png'.format(path, image_index), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        for i in index_array:
            i = int(i.item())
            plt.imshow(image_array[i])
            plt.axis('off')
            plt.savefig(os.path.join(path,'{}.png'.format(i)), bbox_inches='tight',pad_inches = 0)
            plt.close()

def cal_color_difference_left(image_array):
    image_array_left = image_array[927][:,:,:]*255
    image_array_left_2 = image_array_left.copy()
    for i in range(32):
        for j in range(32):
            image_array_left_2[i][j] = colour.convert(image_array_left[i][j],'RGB','CIE Lab')
            image_array_left[i][j] = np.flip(image_array_left[i][j])

    cv.imwrite(r'input_your_savepath\img.png',image_array_left)


#训练网络并输出准确率
if Train_model:
    train_model(net,num_epochs,train_iter,test_iter,loss,optimizer,scheduler,device,num_batches)
#测试网络并输出准确率
else:
    net = torch.load('Model/ResNet_CIFAR_04.pth')
    # confusion_matrix = test_model(net,test_iter,device,output_matrix=True)
    #
    # draw_confusion_matrix(confusion_matrix,'C:/Users/HAHA/Desktop/image_multiply_time6.png')
    image_array = pick_test_image(test_iter,image_label = 6)
    test_picked_image(image_array, image_label = 6)
    # label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # label_list2 = ['飞机','汽车','小鸟','小猫','鹿','小狗','蛙','马','船','卡车']
    # for i in range(len(label_list2)):
    #     output_picked_image(image_array, interval = None, path = r'F:\P-OAAT-CIFAR10images\0{}误判成{}的蛙'.format(i,label_list2[i]),
    #                     image_index=None,index_array=np.loadtxt(r'C:\Users\HAHA\Desktop\false_indexs\false_indexs_{}.csv'.format(label_list[i]),delimiter=','))     # 输出图像前先取消对测试图像的Normalize处理
    output_picked_image(image_array, interval=None,
                        path=r'input_your_savepath',
                        image_index=835,index_array=None)
    # cal_color_difference_left(image_array)                                                                    # 文章图用的青蛙索引为927


