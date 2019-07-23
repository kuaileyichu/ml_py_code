import numpy as np
import pandas as pd

test = [2, 'S']


def Train(features, labels):
    global label, feature
    # 计算结果的类别
    label = np.unique(labels)
    # 计算数据特征的维度,每个维度不止一个特征值
    features_num = features.shape[1]
    feature = np.unique(features)  # 这里可能有文字，文字里面有可能有空格干扰求无重复项
    # 初始化先验概率和条件概率
    prior = np.zeros(len(label))
    conditional = np.zeros([len(label), len(feature)])
    # 首先计算先验概率
    for i in range(len(label)):
        label_class_sum = np.sum(labels == label[i])
        label_len = len(labels)
        prior[i] = float(label_class_sum) / float(label_len)
        # 其次再计算条件概率
        for j in range(len(feature)):
            feature_conditional = features[labels == label[i]]
            feature_class_sum = np.sum(feature_conditional == feature[j])
            conditional[i][j] = float(feature_class_sum) / float(label_class_sum)
    return prior, conditional


def Predict(prior, conditional, test):
    global label, feature
    result = np.zeros(len(label))
    for i in range(len(label)):
        result[i] = conditional[i, feature == test[0]] * conditional[i, feature == test[1]] * (prior[i])
    result = np.vstack([result, label])
    return result


if __name__ == "__main__":
    raw_data = pd.read_csv('D:\\Python27\\yy\\data\\three_bayes.csv')
    raw_data = raw_data.replace(' ', '', regex=True)
    data = raw_data.values
    labels = data[::, 0]
    features = data[::, 1::]
    prior_probability, conditional_probability = Train(features, labels)
    result = Predict(prior_probability, conditional_probability, test)
    print ('the test result is', result)