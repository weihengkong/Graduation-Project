"""
主要负责将数据保存到文件中，或者从文件中读取数据
"""
import json

import numpy as np
import os


def save_to_file(data, filename):
    """
    将二维数组保存到文件中

    参数：
    data：要保存的二维数组
    filename：要保存的文件名

    返回：
    无
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savetxt(filename, data, fmt='%d')


def load_from_file(filename):
    """
    从文件中读取二维数组

    参数：
    filename：要读取的文件名

    返回：
    二维数组
    """
    return np.loadtxt(filename, dtype=int)


# 实现函数存储dict类型的数据到文件中
def save_dict_to_file(data, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'w') as f:
        json.dump(data, f)


# 实现函数从文件中读取dict类型的数据
def load_dict_from_file(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as f:
        data = json.load(f)

    return data


# 使用numpy，实现函数将包含dict的list写入文件
def save_list_to_file(data, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    json.dump(data, open(filename, 'w'))


# 使用numpy，实现函数从文件中读取包含dict的list类型的数据
def load_list_from_file(filename):
    if not os.path.exists(filename):
        return None
    return json.load(open(filename, 'r'))


def isExist(filename):
    """
    判断文件是否存在

    :param filename:
    :return:
    """
    return os.path.exists(filename)
