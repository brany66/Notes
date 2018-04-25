#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by YWJ on 2018/4/20

import numpy as np
import operator

"""
函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力

Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量
"""
def loadFileToMatrix(fileName):
    # open file
    fr = open(fileName)

    # read contents
    contents = fr.readlines()
    # file lines
    numLines = len(contents)
    # construct a init matrix
    mat = np.zeros((numLines, 3))

    labelVector = []
    index = 0

    for line in contents:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()

        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        mat[index, :] = listFromLine[0:3]

        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            labelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            labelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            labelVector.append(3)
        index += 1
    return mat, labelVector
"""
函数说明:对数据进行归一化
Parameters:
	dataSet - 特征矩阵
Returns:
	normDataSet - 归一化后的特征矩阵
	ranges - 数据范围
	minV - 数据最小值
Modify:

"""
def autoNorm(dataSet):
    minV = dataSet.min(0)
    maxV = dataSet.max(0)
    # 最大值和最小值的范围
    ranges = maxV - minV
    # normDataSet = np.zeros(np.shape(dataSet))

    # 返回dataSet的行数
    m = dataSet.shape[0]

    # 原始值减去最小值
    normDataSet = dataSet - np.tile(minV, (m, 1))
    # 除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # 返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minV

"""
函数说明:kNN算法,分类器
Parameters:
	testData - 用于分类的数据(测试集)
	trainData - 用于训练的数据(训练集)
	labels - 分类标签
	K - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果
Modify:
	
"""
def KNN_Classify(testData, trainData, labels, K):
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = trainData.shape[0]

    # 在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(testData, (dataSetSize, 1)) - trainData

    # 二维特征相减后平方
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    # 开方，计算出距离
    distances = ((diffMat ** 2).sum(axis=1) ** 0.5)

    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()

    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(K):
        # 取出前k个元素的类别
        vote_label = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[vote_label] = classCount.get(vote_label, 0) + 1

    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

"""
函数说明:分类器测试函数
Parameters:
	无
Returns:
	normDataSet - 归一化后的特征矩阵
	ranges - 数据范围
	minVals - 数据最小值
"""
def datingClassTest():
    # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    filename = "data.txt"
    datingDataMat, datingLabels = loadFileToMatrix(filename)

    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minV = autoNorm(datingDataMat)

    m = normMat.shape[0] # 获得normMat的行数

    # 取所有数据的百分之十
    numTestVectors = int(m * 0.10 )
    # 分类错误计数
    errorCount = 0.0

    for i in range(numTestVectors):
        # 前numTestVectors个数据作为测试集, 后 m - numTestVectors个数据作为训练集
        classifierResult = KNN_Classify(normMat[i, :], normMat[numTestVectors : m, :],
                                        datingLabels[numTestVectors:m], 4)
        print("分类结果:%s\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0

    print("错误率:%f%%" % (errorCount / float(numTestVectors) * 100))

def classifyPerson():
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征用户输入
    percentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    # 打开的文件名
    filename = "data.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = loadFileToMatrix(filename)
    # 训练集归一化
    normMat, ranges, minV = autoNorm(datingDataMat)

    # 生成NumPy数组,测试集
    inArr = np.array([ffMiles, percentTats, iceCream])
    # 测试集归一化
    norm_inArr = (inArr - minV) / ranges
    # 返回分类结果
    classifierResult = KNN_Classify(norm_inArr, normMat, datingLabels, 3)
    # 打印结果
    print("你可能%s这个人" % (resultList[classifierResult - 1]))

if __name__ == '__main__':

    datingClassTest()
    classifyPerson()





