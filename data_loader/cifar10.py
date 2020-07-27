import os
import pickle
import numpy as npp
from torch.utils.data import Dataset

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='latin1')
        data = data_dict["data"]
        labels = data_dict["labels"]
    return data, np.array(labels)

class CIFAR10TrainSet(Dataset):
    """[加载CIFAR10训练集]

    Args:
        self.dirname ([str]): [CIFAR10数据集所在的文件夹的地址]
        self.is_vector (bool, optional): [加载的CIFAR10数据集是3072向量，还是3*32*32的矩阵]. 
                                        Defaults to False.即matrix
        self.data ([array]): CIFAR10训练数据
        self.labels ([array]): CIFAR10训练标签
    """
    def __init__(self,dirname,is_vector = False):
        """[summary]

        Args:
            dirname ([str]): [CIFAR10数据集所在的文件夹的地址]
            is_vector (bool, optional): [加载的CIFAR10数据集是3072向量，还是3*32*32的矩阵]. 
                                        Defaults to False.即matrix
        """
        self.dirname = dirname
        self.is_vector = True

        data_list = []
        labels_list = []
        train_file = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
        # 打开CIFAR10的五个训练文件
        for file in train_file:
            filename = os.path.join(self.dirname,file)
            batch_datas,batch_labels = unpickle(filename)
            data_list.append(batch_datas)
            labels_list.append(batch_labels)
        # 将CIFAR10的数据整合在一起
        self.data = np.concatenate(data_list,axis=0)
        self.labels = np.concatenate(labels_list,axis=0)

        if not self.is_vector:
            self.data.reshape(-1,3,32,32)     #(50000,3,32,32)
        
    def __getitem__(self, idx):
        return (self.data[idx],self.labels[idx])

    def __len__(self):
        return len(self.data)


class CIFAR10TestSet(Dataset):
    """[加载CIFAR10训练集]

    Args:
        self.dirname ([str]): [CIFAR10数据集所在的文件夹的地址]
        self.is_vector (bool, optional): [加载的CIFAR10数据集是3072向量，还是3*32*32的矩阵]. 
                                        Defaults to False.即matrix
        self.data ([array]): CIFAR10测试数据
        self.labels ([array]): CIFAR10测试标签
    """
    def __init__(self,dirname,is_vector = False):
        """[summary]

        Args:
            dirname ([str]): [CIFAR10数据集所在的文件夹的地址]
            is_vector (bool, optional): [加载的CIFAR10数据集是3072向量，还是3*32*32的矩阵]. Defaults to False.即matrix
        """
        self.dirname = dirname
        self.is_vector = True

        data_list = []
        labels_list = []
        train_file = ['test_batch']
        # 打开CIFAR10的五个训练文件
        for file in train_file:
            filename = os.path.join(self.dirname,file)
            batch_datas,batch_labels = unpickle(filename)
            data_list.append(batch_datas)
            labels_list.append(batch_labels)
        # 将CIFAR10的数据整合在一起
        self.data = np.concatenate(data_list,axis=0)
        self.labels = np.concatenate(labels_list,axis=0)

        if not self.is_vector:
            self.data.reshape(-1,3,32,32)     #(50000,3,32,32)
        
    def __getitem__(self, idx):
        return (self.data[idx],self.labels[idx])

    def __len__(self):
        return len(self.data)