import logging
import cv2
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from sklearn.cluster import KMeans

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


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
    """使用 KMeans 聚类构建视觉词汇, 即 BoW 方法

    Args:
        images (_type_):
        k (int, optional): _description_. Defaults to 50.

    Returns:
        _type_: _description_
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


def train_classifier(data, labels, classifier):
    # 训练分类器
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(data), np.array(labels), test_size=0.2, random_state=42
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # 输出验证集精度
    print(f"Valid Accuracy on the valid set: {accuracy * 100:.2f}%")
    return classifier


def load_model(model_path):
    loaded_model = joblib.load(model_path)
    return loaded_model


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", filename="classifier.log"
    )


    # 构建分类器字典和默认参数
    calssifier_dict = {
        "svm": SVC(kernel="rbf"),
        "logistic": LogisticRegression(max_iter=1000),
        "naive_bayes": MultinomialNB(),
        "decision_tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=10, random_state=42
        ),
        "knn": KNeighborsClassifier(n_neighbors=67),
    }

    # 根据不同特征的 BoW 向量训练分类器，并测试
    for local_feature_type in ["surf","sift","orb"]:
        logging.info(f'{local_feature_type} feature!')

        with open("./food_101/meta/train.txt") as f:
            train_files = f.readlines()
        with open("./food_101/meta/test.txt") as f:
            test_files = f.readlines()

        given_labels = os.listdir("./food_101/images")

        train_files = [
            file.strip() for file in train_files if file.split("/")[0] in given_labels
        ]
        test_files = [
            file.strip() for file in test_files if file.split("/")[0] in given_labels
        ]

        # 得到局部特征和 label
        train_data, train_labels, label_encoder = load_dataset(
            train_files, is_label_encoder=True, local_feature_type=local_feature_type
        )

        # 训练 kmeans 构造词汇表
        stack_train_data = np.vstack(train_data)
        logging.info(f"stack local feature data shape: {stack_train_data.shape}")
        print(stack_train_data.shape)

        try:
            kmeans, visual_vocabulary = build_visual_vocabulary(stack_train_data, k=50)
            joblib.dump(kmeans, f"./kmeans_{local_feature_type}.pkl")

            kmeans = joblib.load(f"./kmeans_{local_feature_type}.pkl")
            bow_des = extract_bow_descriptor(train_data, kmeans, visual_vocabulary)
            np.save(f"./{local_feature_type}_bow_des_train.npy", bow_des)

        except Exception as e:
            logging.error(e)

        for classifier_type in ["svm", "logistic", "naive_bayes", "decision_tree", "knn"]:
            logging.info(f'{classifier_type} classifier!')

            classifier = calssifier_dict[classifier_type]

            # 训练 分类器
            try:
                trained_classifier = train_classifier(
                    bow_des, train_labels, classifier=classifier
                )
                joblib.dump(
                    trained_classifier, f"{classifier_type}_model_{local_feature_type}.pkl"
                )

                # 进行测试
                test_data, test_labels = load_dataset(
                    test_files, is_label_encoder=False, local_feature_type=local_feature_type
                )

                test_bow_des = extract_bow_descriptor(test_data, kmeans, visual_vocabulary)
                np.save(f"./{local_feature_type}_bow_des_test.npy", test_bow_des)

                # 预测结果
                loaded_model = load_model(
                    f"{classifier_type}_model_{local_feature_type}.pkl"
                )
                pred_test_label = loaded_model.predict(test_bow_des)

                # 把测试集的标签转换成数字
                test_label_gt = label_encoder.transform(test_labels)

                # 计算准确性
                accuracy = accuracy_score(test_label_gt, pred_test_label)
                logging.info(f"Test Accuracy on the test set: {accuracy * 100:.2f}%")

            except Exception as e:
                logging.error(e)
