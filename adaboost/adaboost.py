
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


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    # 初始化等分权重值
    D = np.mat(np.ones((m, 1)) / m)
    # 计算每个数据点的类别估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D: ", D.T)
        # 确保程序不会除零溢出
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        expon = np.multiply(-1 * alpha *
                            np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 计算分类的错误率，如果错误率是0，则退出循环
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.transpose())
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(
            classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(dataToClass, classifierArr):
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)
