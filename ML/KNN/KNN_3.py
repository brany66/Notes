#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by YWJ on 2018/4/20

import numpy as np
import operator
from os import listdir

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
函数说明:将32x32的二进制图像转换为1x1024向量。
Parameters:
	filename - 文件名
Returns:
	returnVect - 返回的二进制图像的1x1024向量
Modify:
	2017-03-25
"""
def img2vector(filename):
   returnVector = np.zeros((1, 1024))
   fr = open(filename)

   for i in range(32):
       lineStr = fr.readline()
       #每一行的前32个元素依次添加到returnVector
       for j in range(32):
           returnVector[0, 32*i+j] = int(lineStr[j])

   return returnVector

"""
函数说明:手写数字分类测试
Parameters:
	无
Returns:
	无
"""
def handwritingClassTest():
	#测试集的Labels
	hwLabels = []
	# 返回trainingDigits目录下的文件名
	trainingFileList = listdir('trainingDigits')
	# 返回文件夹下文件的个数
	m = len(trainingFileList)

	#初始化训练的Mat矩阵,测试集
	trainingMat = np.zeros((m, 1024))
	#从文件名中解析出训练集的类别

	for i in range(m):
		#获得文件的名字
		fileNameStr = trainingFileList[i]
		#获得分类的数字
		classNumber = int(fileNameStr.split('_')[0])
		#将获得的类别添加到hwLabels中
		hwLabels.append(classNumber)
		#将每一个文件的1x1024数据存储到trainingMat矩阵中
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

	#返回testDigits目录下的文件名
	testFileList = listdir('testDigits')
	errorCount = 0.0 #错误检测计数
	mTest = len(testFileList) #测试数据的数量

	#从文件中解析出测试集的类别并进行分类测试
	for i in range(mTest):
		#获得文件的名字
		fileNameStr = testFileList[i]
		#获得分类的数字
		classNumber = int(fileNameStr.split('_')[0])
		#获得测试集的1x1024向量,用于训练
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		#获得预测结果
		classifierResult = KNN_Classify(vectorUnderTest, trainingMat, hwLabels, 3)
		print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
		if classifierResult != classNumber:
			errorCount += 1.0

	print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest))

"""
函数说明:main函数
Parameters:
	无
Returns:
	无
Modify:
	2017-03-25
"""
if __name__ == '__main__':
	handwritingClassTest()