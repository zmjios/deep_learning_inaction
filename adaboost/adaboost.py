
import numpy as np


def loadSimpData():
    datMat = np.matrix(
        [[1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
        # 初始化分类结果都为1
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 简历弱分类器，dataArr为输入数据，D为训练的权重矩阵（m*1维）
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    # 变用于在特征的所有可能值上进行遍历,这里看出的数据集中单位最小的是0.1，步长可用1/0.1=10
    numStep = 10.0
    # 存贮决策树
    bestStump = {}
    # 最佳单层决策树的结果，初始化为全部分类错误
    bestClassEst = np.mat(np.zeros((m, 1)))
    # 最小错误，初始化为无穷大
    minError = np.inf
    # 遍历数据集的每一个特征（即为列向量，这里可以看成x和y轴两个特征遍历）
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numStep
        for j in range(-1, int(numStep) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(
                    dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # print("errArr = %s" % errArr)
                weightError = D.T * errArr
                print("split: dim %d, thresh % .2f, thresh inequal: %s, the weighted error is %.3f" % (
                    i, threshVal, inequal, weightError))
                if weightError < minError:
                    minError = weightError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst
