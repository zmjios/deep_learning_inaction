from numpy import *
import operator
from os import listdir


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # shape返回矩阵的魏书 m*n，取出训练集的行数
    # tile是numpy里面重复某个数组的，即将输入inX（inX为1位向量）扩充，在矩阵行上扩展dataSize倍
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2  # 每个元素平方
    sqDistance = sqDiffMat.sum(axis=1)  # 每一列相加
    distance = sqDistance**0.5  # 开平方
    # argsort() 函数将数组的值从小到大排序后，并按照其相对应的索引值输出
    sortedDisIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDisIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化，利用 newValue = （oldVaule - min）/(max - min)

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normalDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalDataSet = dataSet - tile(minVals, (m, 1))
    normalDataSet = normalDataSet / tile(ranges, (m, 1))
    return normalDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]  # 取矩阵维数的行数
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(
            normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with:%d, the real answer is:%d" %
              (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total errot rate is:%f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large does']
    percentTats = float(input("percentage of time spent playing games ?"))
    ffMiles = float(input("frequent filter miles earned per year ?"))
    iceCream = float(input("liters of ice cream consumed year ?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0(
        (inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("you will probably like this person: ",
          resultList[classifierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwrtingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # 名字都是0_10.txt这样的，前面是具体真实的数字，后面是训练集的个数
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        # 名字都是0_10.txt这样的，前面是具体真实的数字，后面是训练集的个数
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back wit: %d, the real answer is: %d" %
              (classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe toatl number of erros is : %d" % errorCount)
    print("\nthe total error rate is:%f" % (errorCount / float(mTest)))
