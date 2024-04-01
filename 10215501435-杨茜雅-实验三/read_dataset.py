import pandas as pd
import numpy as np
class Read_Dataset():
    
    def __init__(self,label1,label2,scores):
        # 初始化函数，接收三个参数：label1、label2、scores
        # 通过调用 return_dataset 函数读取数据集，并将训练集、验证集和测试集保存在实例变量中
        self.train_dataset,self.dev_dataset,self.test_dataset = return_dataset(label1,label2,scores)
    
    def get_set(self,train):
        # 根据传入的参数 train，返回对应的数据集（训练集和验证集或测试集）
        if train == "train":
            return self.train_dataset,self.dev_dataset
        elif train == "test":
            return self.test_dataset

def return_dataset(label1,label2,scores):
    # 辅助函数，用于读取数据集并返回训练集、验证集和测试集
    dtype = [(label1, np.int32), (label2, np.int32), (scores, np.float32)]
    # 定义一个 dtype，指定各列的名称和数据类型
    print("正在读取数据......")
    print("正在读取训练集数据......")
    train_dataset = pd.read_csv("data/train.csv", usecols=range(3), dtype=dict(dtype))
    # 使用 pandas 库的 read_csv 函数读取训练集数据，仅使用前三列，并指定数据类型为 dtype
    print("正在读取验证集数据......")
    dev_dataset = pd.read_csv("data/dev.csv", usecols=range(3), dtype=dict(dtype))
    # 使用 pandas 库的 read_csv 函数读取验证集数据，仅使用前三列，并指定数据类型为 dtype
    print("正在读取测试集数据......")
    dtype = [("userId", np.int32), ("movieId", np.int32)]
    test_dataset = pd.read_csv("data/test.csv", usecols=range(1,3), dtype=dict(dtype))
     # 使用 pandas 库的 read_csv 函数读取测试集数据，仅使用第二列和第三列，并指定数据类型为 dtype
    print("数据载入完毕")
    return train_dataset,dev_dataset,test_dataset
    # 返回读取的训练集、验证集和测试集数据