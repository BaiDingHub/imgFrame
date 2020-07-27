import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader



class MnistTrainSet(Dataset):
    """[加载Mnist训练集]

    Args:
        self.dirname ([str]): [CIFAR10数据集所在的文件夹的地址]
        self.is_vector (bool, optional): [加载的CIFAR10数据集是3072向量，还是3*32*32的矩阵]. 
                                        Defaults to False.即matrix
        self.data ([array]): CIFAR10训练数据
        self.labels ([array]): CIFAR10训练标签
    """
    def __init__(self,dirname,is_vector = True):
        """[summary]

        Args:
            dirname ([str]): [Mnist数据集所在的文件夹的地址]
            is_vector (bool, optional): [加载的Mnist数据集是784向量，还是28*28的矩阵]. 
                                        Defaults to True.即vector
        """
        self.dirname = dirname
        self.is_vector = True
        # 得到路径
        train_img_path = os.path.join(dirname,'train-images-idx3-ubyte')
        train_label_path = os.path.join(dirname,'train-labels-idx1-ubyte')
        # 加载图片
        self.data = load_mnist_data(train_img_path)           #(60000,784)
        self.labels = load_mnist_label(train_label_path)     #(60000,)
        # 如果需要将数据集转换为图片
        if not is_vector:
            # 将图片转换为(60000,1,28,28)
            self.data = self.data.reshape(-1,1,28,28)  

    def __getitem__(self, idx):
        return (self.data[idx],self.labels[idx])

    def __len__(self):
        return len(self.data)



class MnistTestSet(Dataset):
    """[加载Mnist测试集]

    Args:
        self.dirname ([str]): [CIFAR10数据集所在的文件夹的地址]
        self.is_vector (bool, optional): [加载的CIFAR10数据集是3072向量，还是3*32*32的矩阵]. 
                                        Defaults to False.即matrix
        self.data ([array]): CIFAR10测试数据
        self.labels ([array]): CIFAR10测试标签
    """
    def __init__(self,dirname,is_vector = True):
        """[summary]

        Args:
            dirname ([str]): [Mnist数据集所在的文件夹的地址]
            is_vector (bool, optional): [加载的Mnist数据集是784向量，还是28*28的矩阵]. 
                                        Defaults to True.即vector
        """
        self.dirname = dirname
        self.is_vector = True
        # 得到路径
        test_img_path = os.path.join(dirname,'t10k-images-idx3-ubyte')
        test_label_path = os.path.join(dirname,'t10k-labels-idx1-ubyte')
        # 加载图片
        self.data = load_mnist_data(test_img_path)         #(10000,784)
        self.labels = load_mnist_label(test_label_path)     #(10000,)
        # 如果需要将数据集转换为图片
        if not is_vector:
            # 将图片转换为(60000,1,28,28)
            self.data = self.data.reshape(-1,1,28,28)  
        
    def __getitem__(self, idx):
        return (self.data[idx],self.labels[idx])


    def __len__(self):
        return len(self.data)

    

def load_mnist_data(filename):
    """[加载Mnist's data]

    Args:
        filename ([str]): [Mnist图片所在的文件夹]

    Returns:
        imgs [type]: [(60000,784))]
    """
    binfile = open(filename, 'rb') # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0) # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bits_string = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bits_string, buffers, offset) # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height]) # reshape为[60000,784]型数组
    imgs = imgs/255

    return imgs


def load_mnist_label(filename):
    """[加载Mnist's label]

    Args:
        filename ([str]): [Mnist‘s label所在的文件夹]

    Returns:
        labels [array]: [(60000,)]
    """
    binfile = open(filename, 'rb') # 读二进制文件
    buffers = binfile.read()
 
    head = struct.unpack_from('>II', buffers, 0) # 取label文件前2个整形数
 
    label_num = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置
 
    string_num = '>' + str(label_num) + "B" # fmt格式：'>60000B'
    labels = struct.unpack_from(string_num, buffers, offset) # 取label数据
 
    binfile.close()
    labels = np.reshape(labels, [label_num]) # 转型为列表(一维数组)
 
    return labels





