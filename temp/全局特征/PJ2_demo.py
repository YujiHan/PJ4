import cv2
import numpy as np
from scipy.fftpack import dct

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# 加载数据集
def load_data_food101(list_path):
    # 初始化存储图像路径和标签的列表
    img_paths = []
    labels = []

    # 将食物名称映射到整数标签
    label_str_to_int = {
        'sushi': 1,
        'tacos': 2,
        'takoyaki': 3,
        'tiramisu': 4,
        'tuna_tartare': 5,
        'waffles': 6,
    }

    # 打开包含图像信息的文件并读取每一行
    with open(list_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # 处理标签
        label_str, _ = line.split('/')  # 分割行以获取图像的标签部分
        label_int = label_str_to_int.get(label_str)  # 转换为数字表示的标签
        labels.append(label_int)  # 将数字标签添加到标签列表

        # 构建图像的完整路径
        img_path = (
            '/Users/hanyuji/codes/draft_demo/CV_PJ2/food_101/images/'
            + line.strip()
            + '.jpg'
        )
        img_paths.append(img_path)  # 将图像路径添加到路径列表
    return img_paths, labels  # 返回图像路径和对应的标签列表


# CLD
def cld_features(image_path):
    image = cv2.imread(image_path)  # 读取图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)  # 转换到 YCrCb 颜色空间
    image = cv2.resize(image, (64, 64))  # 缩放图像到64x64像素

    # 分割图像，计算64x64图像中每个8x8小块的平均颜色
    tiles_avg_color = np.zeros((8, 8, 3))
    for x in range(0, 64, 8):
        for y in range(0, 64, 8):
            tile = image[x : x + 8, y : y + 8]
            for channel in range(3):
                tiles_avg_color[int(x / 8), int(y / 8), channel] = np.mean(
                    tile[:, :, channel]
                )

    # 应用 DCT离散余弦变换，分3个通道分别进行
    dct_coeffs = np.zeros((8, 8, 3))
    dct_coeffs[:, :, 0] = dct(tiles_avg_color[:, :, 0])
    dct_coeffs[:, :, 1] = dct(tiles_avg_color[:, :, 1])
    dct_coeffs[:, :, 2] = dct(tiles_avg_color[:, :, 2])

    # 将dct系数 按照之字形展开
    zig = [
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ]
    dct_coeffs_flatten = np.zeros((64, 3))
    for i in range(8):
        for j in range(8):
            dct_coeffs_flatten[zig[i][j], :] = dct_coeffs[i, j, :]

    return dct_coeffs_flatten.flatten()


# SCD
def scd_features(image_path, num_bins=16):
    image = cv2.imread(image_path)  # 读取图像
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换到HSV颜色空间

    # 计算HSV颜色空间中每个通道的直方图
    hist = cv2.calcHist(
        [image_hsv], [0, 1, 2], None, [num_bins] * 3, [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()  # 归一化直方图并将其扁平化

    # 应用DCT离散余弦变换以减少特征的维度
    dct_features = dct(dct(hist, norm='ortho'), norm='ortho')
    return dct_features


# 颜色直方图特征
def histogram_features(image_path, bins=8):
    # 读取图像并转换到 HSV 颜色空间
    image = cv2.imread(image_path)  # 读取图像
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换到HSV颜色空间

    # 计算HSV颜色空间中每个通道的直方图
    # 对于每个通道设置相同数量的bins，并定义直方图的范围
    hist = cv2.calcHist(
        [image_hsv], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]
    )
    # 归一化直方图并将其扁平化
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def features_and_classifier(features, classifier):
    # 加载训练集数据和标签
    image_paths_train, labels_train = load_data_food101(
        '/Users/hanyuji/codes/draft_demo/CV_PJ2/food_101/meta/train.txt'
    )
    # 加载测试集数据和标签
    image_paths_test, labels_test = load_data_food101(
        '/Users/hanyuji/codes/draft_demo/CV_PJ2/food_101/meta/test.txt'
    )

    # 提取特征
    features_train = np.array([features(path) for path in image_paths_train])
    features_test = np.array([features(path) for path in image_paths_test])

    # 对特征进行绝对值处理，以保证特征值为非负
    features_train = np.abs(features_train)
    features_test = np.abs(features_test)

    classifier.fit(features_train, labels_train)  # 使用提供的分类器在训练集上进行训练

    # 在测试集上评估分类器
    accuracy = classifier.score(features_test, labels_test)
    print(f"分类准确率: {accuracy}")

    # predict = classifier.predict(features_test)
    # print(classification_report(labels_test, predict, labels=[-1, 1]))


def main():
    # 特征提取方法
    features_list = {
        'CLD': cld_features,
        'SCD': scd_features,
        'hist': histogram_features,
    }

    # 分类器
    classifier = {
        "svm": SVC(kernel="rbf"),
        "logistic": LogisticRegression(max_iter=1000),
        "naive_bayes": MultinomialNB(),
        "decision_tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=10, random_state=42
        ),
        "knn": KNeighborsClassifier(n_neighbors=67),
    }

    # 循环遍历所有特征提取方法和分类器的组合
    for key_feature, v_feature in features_list.items():
        for key_classifier, v_classifier in classifier.items():
            print(key_feature + '---' + key_classifier + ':')
            features_and_classifier(v_feature, v_classifier)  # 调用函数来测试特定的特征提取方法和分类器
            print('--------------------')


if __name__ == '__main__':
    main()
