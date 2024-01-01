import cv2
import os
import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


def extract_features(image_path, local_feature_type: str = "sift"):
    # 使用 opencv 提取局部特征的描述子
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

# 步骤2: 构建数据集和标签
def load_dataset(train_files, is_label_encoder=True, local_feature_type: str = "sift"):
    data = []
    labels = []
    image_path = "./food_101/images/"
    for file_name in train_files:
        file_path = os.path.join(image_path, file_name + ".jpg")
        # 提取每张图片的局部特征
        features = extract_features(file_path, local_feature_type=local_feature_type)
        if len(features) > 0:
            data.append(features)
            label = file_name.split("/")[0]
            labels.append(label)
    # 如果需要对标签进行编码
    if is_label_encoder:
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        return data, labels, label_encoder

    return data, labels


def build_visual_vocabulary(data, k=50):
    """
    使用 KMeans 聚类构建视觉词汇, 即 BoW 方法
    """
    # 使用 KMeans 聚类形成视觉词汇
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    visual_vocabulary = kmeans.cluster_centers_

    return kmeans, visual_vocabulary


def extract_bow_descriptor(data, kmeans, visual_vocabulary):
    # 讲描述子转化为 BoW 向量
    bow_descriptors = []

    tq = tqdm(data)
    for descriptor in tq:
        # 使用 KMeans 模型将每个描述子映射到最近的视觉词汇
        labels = kmeans.predict(descriptor)

        # 统计每个视觉词汇的出现频率
        bow_descriptor = np.bincount(labels, minlength=len(visual_vocabulary))

        # 归一化
        bow_descriptor = bow_descriptor / np.sum(bow_descriptor)

        bow_descriptors.append(bow_descriptor)

    return bow_descriptors

