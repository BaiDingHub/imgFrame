import torch
import numpy as np
import os
from torch.utils.data import Dataset,DataLoader
import struct


class MnistTrainSet(Dataset):
    """[加载Mnist训练集]
    """
    def __init__(self,dirname,needVector = True):
        """[summary]

        Args:
            dirname ([str]): [Mnist数据集所在的文件夹的地址]
            needVector (bool, optional): [加载的Mnist数据集是784向量，还是28*28的矩阵]. Defaults to True.即vector
        """
        self.dirname = dirname
        self.needVector = True
        #得到路径
        trainImgPath = os.path.join(dirname,'train-images-idx3-ubyte')
        trainLabelPath = os.path.join(dirname,'train-labels-idx1-ubyte')
        #加载图片
        self.trainX = loadMnistImg(trainImgPath)         #(60000,784)
        self.trainY = loadMnistLabel(trainLabelPath)     #(60000,)
        #如果需要将数据集转换为图片
        if not needVector:
            self.trainX = self.trainX.reshape(-1,1,28,28)  #将图片转换为(60000,1,28,28)
        

    def __getitem__(self, idx):
        return (self.trainX[idx],self.trainY[idx])

    def __len__(self):
        return len(self.trainX)



class MnistTestSet(Dataset):
    """[加载Mnist测试集]
    """
    def __init__(self,dirname,needVector = True):
        """[summary]

        Args:
            dirname ([str]): [Mnist数据集所在的文件夹的地址]
            needVector (bool, optional): [加载的Mnist数据集是784向量，还是28*28的矩阵]. Defaults to True.即vector
        """
        self.dirname = dirname
        self.needVector = True
        #得到路径
        testImgPath = os.path.join(dirname,'t10k-images-idx3-ubyte')
        testLabelPath = os.path.join(dirname,'t10k-labels-idx1-ubyte')
        #加载图片
        self.testX = loadMnistImg(testImgPath)         #(10000,784)
        self.testY = loadMnistLabel(testLabelPath)     #(10000,)
        #如果需要将数据集转换为图片
        if not needVector:
            self.testX = self.testX.reshape(-1,1,28,28)  #将图片转换为(60000,1,28,28)
        

    def __getitem__(self, idx):
        return (self.testX[idx],self.testY[idx])


    def __len__(self):
        return len(self.testX)

    

def loadMnistImg(filename):
    '''
    输入
    filename: Mnist图片所在的路径
    输出
    imgs：(60000,784)
    '''
    binfile = open(filename, 'rb') # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0) # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset) # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height]) # reshape为[60000,784]型数组
    imgs = imgs/255

    return imgs


def loadMnistLabel(filename):
    '''
    输入
    filename: Mnist's Label所在的路径
    输出
    labels:(60000,)
    '''
    binfile = open(filename, 'rb') # 读二进制文件
    buffers = binfile.read()
 
    head = struct.unpack_from('>II', buffers, 0) # 取label文件前2个整形数
 
    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置
 
    numString = '>' + str(labelNum) + "B" # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset) # 取label数据
 
    binfile.close()
    labels = np.reshape(labels, [labelNum]) # 转型为列表(一维数组)
 
    return labels





