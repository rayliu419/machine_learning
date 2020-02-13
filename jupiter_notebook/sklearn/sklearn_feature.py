# encoding=utf-8

'''
UCI薪水预测问题
效果不断优化
对feature做各种处理，比较对模型的提升效果
在使用训练集时，一定要注意数据文件格式是不是有什么问题！！！
'''

import sklearn.preprocessing
from pylsy import *
import numpy as np
from sklearn_model import *
import math
import matplotlib as plt
import pandas as pd


#############################################################
#  观察数据
#############################################################

def draw_distribution(original_data):
    fig = plt.figure(figsize=(20, 15))
    cols = 5
    rows = math.ceil(float(original_data.shape[1]) / cols)
    for i, column in enumerate(original_data.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if original_data.dtypes[column] == np.object:
            original_data[column].value_counts().plot(kind="bar", axes=ax)
        else:
            original_data[column].hist(axes=ax)
            plt.xticks(rotation="vertical")
    plt.subplots_adjust(hspace=0.7, wspace=0.2)
    plt.show()


def print_basic_information(df):
    # 多少个sample，多少个属性
    print "rows : {0}  columns : {1}".format(df.shape[0], df.shape[1])
    # 属性名称
    print df.columns
    # 对于数值属性，打印平均值，方差，min，百分位数等
    print df.describe()
    print df.dtypes
    print df["target"].value_counts()
    # df.hist()
    # plt.show()
    # print df.cov()
    # print df.corr()


# 打印数据中各个列里最top的那些值00
def print_top_in_col(original_data, select_col_names, all_names):
    if not set(select_col_names).issubset(set(all_names)):
        print('some invalid column names in select_col_names')
        return
    for item in select_col_names:
        print('column : ' + item)
        print((original_data[item].value_counts() / original_data.shape[0]).head())
        print('\n')


#############################################################
#  数据集的特殊处理
#############################################################

def filter_record(df, check_col_names):
    '''
    过滤掉check_col_names的na值
    '''
    original_data = df.dropna(subset=check_col_names)
    return original_data


#############################################################
#  数值化方法
#############################################################

def enum_feature_transform(df):
    '''
    属性数值化
    实际应用中，应该为每个feature选择数值化的方法。
    这里使用的统一编码方式
    :return:
    '''
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:  # 对于非数值属性
            # 简单编码方式，即枚举所有类型，按0,1,2,3...编码
            encoders[column] = sklearn.preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


def dummy_feature_transform(df):
    '''
    属性数值化
    dummpy编码方式
    '''
    result = pd.get_dummies(df)
    return result


def compare_different_feature_numeration(df):
    '''
    比较不同的数值化的编码效果
    :param df:
    :return:
    '''
    temp = df[0:5]
    temp = temp.loc[:, ["age", "workclass", "fnlwgt", "education", "target"]]
    print "======================================================================="
    print temp
    print "======================================================================="
    result, _ = enum_feature_transform(temp)
    print result
    print "======================================================================="
    result = dummy_feature_transform(temp)
    print result


#############################################################
#  归一化
#############################################################

def standard_scaler(df):
    scalar = sklearn.preprocessing.StandardScaler()
    return_df = scalar.fit_transform(df)
    return return_df


def compare_different_scaler(df):
    '''
    比较不同的归一化效果
    :param df:
    :return:
    '''
    temp = df[0:2]
    print "======================================================================="
    print temp
    result = standard_scaler(temp)
    print "======================================================================="
    print result


#############################################################
#  离散化
#############################################################

def age_discretization(age):
    if (age < 22):
        return "1"
    elif (age < 28):
        return "2"
    elif (age < 35):
        return "3"
    elif (age < 50):
        return "4"
    elif (age < 60):
        return "5"
    else:
        return "6"


#############################################################
#  切分数据集，切分属性和预测变量
#############################################################

def split_train_test_seq(df, size):
    '''
    简单分割法
    '''
    return df[0:size], df[size:]


def split_X_Y(df, Y_names):
    '''
    分离feature和预测变量, Y用Y_name标示
    '''
    Y = df.loc[:, Y_names]
    X = df.drop(Y_names, axis=1)
    return X, Y


#############################################################
#  读取数据
#############################################################

def load_data(training_set_file, validation_set_File):
    '''
    注意read_table和read_csv默认第一行是放了列名称的，它不是以names来放入key的，
    而是用读到的第一行来加入到dataframes.columns
    可以用header=None来去掉

    '''
    training_data = pd.read_table(training_set_file, engine="python", na_values="?",
                                  delimiter=r"\s*,\s")
    validate_data = pd.read_table(validation_set_File, engine="python", na_values="?",
                                  delimiter=r"\s*,\s")
    return training_data, validate_data


#############################################################
#  性能表
#############################################################

def run_model_and_evaluate(X, Y, standardize=False):
    attributes = ["indicator", "LR", "AB", "GNB", "RF"]
    table = pylsytable(attributes)
    table.add_data("indicator", ["f1", "precision", "recall"])
    name, f1, precision, recall = evaluate_model_k_fold("LR", X, Y, standardize)
    table.add_data(name, [str(f1), str(precision), str(recall)])
    name, f1, precision, recall = evaluate_model_k_fold("GNB", X, Y, standardize)
    table.add_data(name, [str(f1), str(precision), str(recall)])
    # 对于svm，没有归一化的都收敛不了
    name, f1, precision, recall = evaluate_model_k_fold("AB", X, Y, standardize)
    table.add_data(name, [str(f1), str(precision), str(recall)])
    name, f1, precision, recall = evaluate_model_k_fold("RF", X, Y, standardize)
    table.add_data(name, [str(f1), str(precision), str(recall)])

    '''
    name, f1, precision, recall = evaluate_model_k_fold("SVC", X, Y, standardize)
    table.add_data(name, [str(f1), str(precision), str(recall)])
    '''
    print table


#############################################################
#  主函数
#############################################################

if __name__ == "__main__":
    training_set_file = "data/adult/adult.data"
    test_set_file = "data/adult/adult.test"
    training_data, test_set_file = load_data(training_set_file, test_set_file)
    col_names = training_data.columns
    training_raw = filter_record(training_data, col_names)
    test_set_raw = filter_record(test_set_file, col_names)
    total_raw = training_raw.append(test_set_raw)
    print total_raw.shape

    # 1. 打印数据基本信息

    print "训练集数据信息："
    print_basic_information(training_raw)
    print "======================================================================="
    print "测试集数据信息："
    print_basic_information(test_set_raw)
    print "======================================================================="

    # 2. 不同的数值化方法效果比较 - 用int值编码
    print "int encoding"
    encode_total_data, _ = enum_feature_transform(total_raw)
    # 要使用kfold的方式评估模型，需要将index names重新编号，因为前面已经去除了某些不合法行
    encode_total_data.reset_index(inplace=True)
    X, Y = split_X_Y(encode_total_data, ["target"])
    run_model_and_evaluate(X, Y)

    print "dummy encoding"
    encode_total_data = dummy_feature_transform(total_raw)
    encode_total_data.reset_index(inplace=True)
    X, Y = split_X_Y(encode_total_data, ["target_<=50K", "target_>50K"])
    Y_1 = Y["target_<=50K"]
    Y_1.columns = ["target"]
    run_model_and_evaluate(X, Y_1)

    # 3. 加入数据的归一化，对于某些模型应该有很明显的提升效果，尤其是LR
    # 做完归一化以后，发现dataframe变成了ndarray了
    # 对于训练集和测试集应该统一的做归一化，不能做不一样的归一化，因为归一化依赖于全局的均值，方差等
    print "dummy encoding + feature_standardization"
    X_standard = standard_scaler(X)
    run_model_and_evaluate(X_standard, Y_1, standardize=True)

    # 4.对数据不是做统一的归一化，而是部分的归一化，分开做一下，
    # 然后把定制化的归一化属性插入到转换后的dataframe中
    print "dummy encoding + feature_standardization + customize feature discretization"
    total_df = total_raw.copy()
    age = total_df["age"]
    total_df = total_df.drop("age", axis=1)
    age = age.map(age_discretization)
    total_df["age"] = pd.Series(age, index=total_df.index)
    encode_total_data = dummy_feature_transform(total_df)
    encode_total_data.reset_index(inplace=True)
    X, Y = split_X_Y(encode_total_data, ["target_<=50K", "target_>50K"])
    X_standard = standard_scaler(X)
    Y_1 = Y["target_<=50K"]
    Y_1.columns = ["target"]
    run_model_and_evaluate(X_standard, Y_1, standardize=True)

    # 5.加入feature组合

    # 6.去除某些feature
    print "dummy coding + reduce feature by random"
    encode_total_data = dummy_feature_transform(total_raw)
    encode_total_data.reset_index(inplace=True)
    X, Y = split_X_Y(encode_total_data, ["target_<=50K", "target_>50K"])
    Y_1 = Y["target_<=50K"]
    Y_1.columns = ["target"]
    X = X.drop("age", axis=1)
    run_model_and_evaluate(X, Y_1)
