import numpy as np
import cv2
import os
from scipy.fftpack import dct


# 加载数据集
def load_dataset(list_path, global_feature_type="CLD"):
    """
    加载数据集
    输入：训练集/测试集图片列表的路径
            "./data/food_101/meta/test.txt" or "./data/food_101/meta/train.txt"
        全局特征类型（CLD，SCD，HIST）
    输出：图片的全局特征向量
        标签（0 - 5）
        标签的encoder，用于转换标签
    """

    with open(list_path, 'r') as f:
        train_files = f.readlines()

    train_files = [file.strip() for file in train_files]

    data = []
    labels = []
    image_path = "./data/food_101/images/"

    for index, file_name in enumerate(train_files):
        print(f'load image: {index}')
        file_path = os.path.join(image_path, file_name + ".jpg")

        # 提取每张图片的局部特征
        features = extract_features(file_path, global_feature_type=global_feature_type)

        if len(features) > 0:
            data.append(features)
            label = file_name.split("/")[0]
            labels.append(label)

    # 将食物名称映射到整数标签
    label_encoder = {
        'sushi': 1,
        'tacos': 2,
        'takoyaki': 3,
        'tiramisu': 4,
        'tuna_tartare': 5,
        'waffles': 6,
    }
    labels = [label_encoder[label] for label in labels]

    return data, labels


# 三种全局特征
def extract_features(image_path, global_feature_type="CLD"):
    """
    提取全局特征
    输入：单张的图片路径
        全局特征类型（CLD，SCD，HIST）
    输出：图片的全局特征向量
    """

    if global_feature_type == "CLD":
        features = cld_features(image_path)
    elif global_feature_type == "SCD":
        features = scd_features(image_path)
    elif global_feature_type == "HIST":
        features = histogram_features(image_path)

    normalized_features = (features - features.min(axis=0)) / (
        features.max(axis=0) - features.min(axis=0)
    )
    return normalized_features


# CLD
def cld_features(image_path):
    """
    提取CLD特征
    输入：单张的图片路径
    输出：图片的CLD特征
    """
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
def scd_features(image_path):
    """
    提取SCD特征
    输入：单张的图片路径
    输出：图片的SCD特征
    """
    image = cv2.imread(image_path)

    # 将图像转换到HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 计算颜色直方图（假设我们使用每个通道的8个bin）
    color_hist = cv2.calcHist(
        [hsv_image], [0, 1, 2], None, [6, 6, 6], [0, 180, 0, 256, 0, 256]
    )
    color_hist = cv2.normalize(color_hist, color_hist).flatten()

    # 边缘检测
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)

    # 计算边缘方向
    sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=5)
    direction = np.arctan2(sobely, sobelx) * 180 / np.pi

    # 计算方向直方图（假设我们使用16个bin）
    direction_hist, _ = np.histogram(direction, bins=16, range=(-180, 180))
    direction_hist = direction_hist.astype("float")
    direction_hist /= direction_hist.sum() + 1e-6

    # 合并特征向量
    scd_feature = np.hstack([color_hist, direction_hist])

    return scd_feature


# HIST
def histogram_features(image_path, bins=6):
    """
    提取颜色直方图特征
    输入：单张的图片路径
    输出：图片的颜色直方图特征
    """
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
