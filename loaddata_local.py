import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    filename='/home/hanyuji/projects/PJ4/local.log',
)


# 加载数据集
def load_dataset(list_path, local_feature_type="sift"):
    """
    加载数据集
    输入：训练集/测试集图片列表的路径
            "./data/food_101/meta/test.txt" or "./data/food_101/meta/train.txt"
        局部特征类型（sift，orb，surf）
    输出：图片的局部特征向量
        标签（0 - 5）
        标签的encoder，用于转换标签
    """

    with open(list_path) as f:
        dataset_files = f.readlines()

    dataset_files = [file.strip() for file in dataset_files]

    data = []
    labels = []

    image_path = "/home/hanyuji/projects/PJ4/data/food_101/images/"
    for index, file_name in enumerate(dataset_files):
        print(f'load image: {index}')
        file_path = os.path.join(image_path, file_name + ".jpg")
        # 提取每张图片的局部特征
        features = extract_features(file_path, local_feature_type=local_feature_type)
        if len(features) > 0:
            data.append(features)
            label = file_name.split("/")[0]
            labels.append(label)
            logging.info(f'index: {index} ---- shape: {features.shape}')

        else:
            label = file_name.split("/")[0]
            labels.append(label)
            logging.info(f'-----------------error-----{index}---------{file_name}')

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

    # 训练 kmeans 构造词汇表
    stack_train_data = np.vstack(data)
    kmeans, visual_vocabulary = build_visual_vocabulary(stack_train_data, k=50)
    bow_desc = extract_bow_descriptor(data, kmeans, visual_vocabulary)

    return bow_desc, labels


# 三种局部特征
def extract_features(image_path, local_feature_type="sift"):
    """
    使用 opencv 提取局部特征的描述子
    输入：单张的图片路径
        局部特征类型（sift，orb，surf）
    输出：图片的descriptors，维度为2
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if local_feature_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
    elif local_feature_type == "orb":
        orb = cv2.ORB_create()
        keypoints = orb.detect(img, None)
        keypoints, descriptors = orb.compute(img, keypoints)
    elif local_feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create(400)
        keypoints, descriptors = surf.detectAndCompute(img, None)
    return descriptors if descriptors is not None else np.array([])


# 将局部特征聚类
def build_visual_vocabulary(data, k=50):
    """
    使用 KMeans 聚类构建视觉词汇, 即 BoW 方法
    """

    # 使用 KMeans 聚类形成视觉词汇
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    visual_vocabulary = kmeans.cluster_centers_

    return kmeans, visual_vocabulary


# 讲描述子转化为 BoW 向量
def extract_bow_descriptor(data, kmeans, visual_vocabulary):
    bow_descriptors = []

    for descriptor in data:
        # 使用 KMeans 模型将每个描述子映射到最近的视觉词汇
        labels = kmeans.predict(descriptor)

        # 统计每个视觉词汇的出现频率
        bow_descriptor = np.bincount(labels, minlength=len(visual_vocabulary))
        # 归一化
        bow_descriptor = bow_descriptor / np.sum(bow_descriptor)
        bow_descriptors.append(bow_descriptor)

    return np.array(bow_descriptors)
