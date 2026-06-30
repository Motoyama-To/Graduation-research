# 猫+犬 品種分類・類似画像検索プログラム

# --- モデル全体の処理 ---
# 入力画像
#   ↓
# YOLOv8で猫・犬を検出
#   ↓ detect_and_crop_animal()
# データ拡張
# （回転・ズーム・明るさ変更・反転・せん断・平行移動）
#   ↓ apply_data_augmentation()
# 特徴量抽出
#   ├─ ResNet18：画像全体の特徴
#   ├─ Color Histogram：毛色・色分布
#   └─ Grid Local Features
#       ├─ HSVヒストグラム
#       ├─ LBP（模様・質感）
#       └─ Edge Density（輪郭情報）
#   ↓ extract_features()
# 特徴量を結合
#   ↓
# StandardScalerで標準化
#   ↓ prepare_data()
# PCAで次元削減（情報量95%を保持）
#   ↓
# Random Forestで学習・分類
#   ↓ train_and_evaluate_model()
# 品種分類結果（猫10種・犬10種）
#   ↓
# 学習済みモデル・PCA・Scalerを保存

# YOLO + ResNet + Color + Grid + PCA + RF

# train.py

import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib # モデル保存のために追加
from skimage.feature import local_binary_pattern # LBPのために追加
from sklearn.metrics.pairwise import cosine_similarity # コサイン類似度のために追加
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms


yolo_model = YOLO("yolov8n.pt")

# --- Matplotlibの日本語表示設定 ---
try:
    plt.rcParams['font.family'] = 'Meiryo'
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

# --- 定数 ---
IMAGE_SIZE = (128, 128)
DATA_DIR = '20-100images'
RANDOM_STATE = 42


# --- 猫クラス ---
cat_classes = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Sphynx"
]

# --- 犬クラス ---
dog_classes = [
    "american_bulldog",
    "beagle",
    "Boxer",
    "chihuahua",
    "english_setter",
    "german_shorthaired",
    "japanese_chin",
    "leonshond",
    "samoyed",
    "shiba_inu"
]

cat_class_names = sorted(cat_classes)
dog_class_names = sorted(dog_classes)

print(os.getcwd())



cropped_cache = {}
CROP_DIR = "cropped_cache"
os.makedirs(CROP_DIR, exist_ok=True)

FEATURE_DIR = "feature_cache"
os.makedirs(FEATURE_DIR, exist_ok=True)


cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- YOLOクロップ関数 ---
def detect_and_crop_animal(image, yolo_model):
    results = yolo_model(image, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return image, "unknown"

    max_area = 0
    best_crop = image
    detected_label = "unknown"

    h, w = image.shape[:2]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]

        if label not in ["cat", "dog"]:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 不正bbox除外
        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)

        if area > max_area:
            max_area = area
            best_crop = image[y1:y2, x1:x2]
            detected_label = label

    return best_crop, detected_label


# --- ResNet18 CNN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)

# 最後の分類層(fc)を削除
resnet = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

resnet.eval()
resnet.to(device)


# --- 元画像パス収集（リーク防止用） ---
def collect_image_paths(data_dir):
    filepaths = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                filepaths.append(img_path)
                labels.append(class_idx)

    return np.array(filepaths), np.array(labels), class_names

# --- データ拡張関数 ---
def apply_data_augmentation(image):
    augmented_images = [image]
    # rotate
    for angle in [-10, 10]:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        augmented_images.append(rotated)
    for _ in range(2):
        zoom_factor = random.uniform(0.95, 1.05)
        h, w = image.shape[:2]
        # zoom in 
        if zoom_factor <= 1.0:
            zh, zw = int(h * zoom_factor), int(w * zoom_factor)
            start_h, start_w = (h - zh) // 2, (w - zw) // 2
            cropped = image[start_h:start_h+zh, start_w:start_w+zw]
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            augmented_images.append(zoomed)
        # zoom out
        else:
            zh, zw = int(h / zoom_factor), int(w / zoom_factor)

            zoomed_out = cv2.resize(
                image, 
                (zw, zh), 
                interpolation=cv2.INTER_LINEAR
            )
            pad_h = h - zh
            pad_w = w - zw

            top = pad_h // 2
            bottom = pad_h - top

            left = pad_w // 2
            right = pad_w - left

            padded = cv2.copyMakeBorder(
                zoomed_out,
                top,
                bottom,
                left,
                right,
                borderType=cv2.BORDER_REFLECT_101
            )

            augmented_images.append(padded)

    # bright
    for _ in range(2):
        factor = random.uniform(0.8, 1.2)
        bright = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        augmented_images.append(bright)
    
    # Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # Horizontal shear
    shear_factor = random.uniform(-0.1, 0.1)
    rows, cols = image.shape[:2]
    M_shear = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(image, M_shear, (cols, rows), borderMode=cv2.BORDER_REFLECT_101)
    augmented_images.append(sheared)
    
    # Vertical shear
    M_shear_v = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
    sheared_v = cv2.warpAffine(image, M_shear_v, (cols, rows), borderMode=cv2.BORDER_REFLECT_101)
    augmented_images.append(sheared_v)
    
    # shift
    for _ in range(2):
        max_shift = 0.05
        tx = int(random.uniform(-max_shift, max_shift) * cols)
        ty = int(random.uniform(-max_shift, max_shift) * rows)
        M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(image, M_shift, (cols, rows), borderMode=cv2.BORDER_REFLECT_101)
        augmented_images.append(shifted)
    return augmented_images

    
def extract_cnn_features(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = cnn_transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        features = resnet(tensor)

    features = features.squeeze().cpu().numpy()

    return features


# --- 特徴量抽出関数 ---
def extract_features(image):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CNN 全体特徴
    cnn_features = extract_cnn_features(image)
    
    
    # Color Histogram  毛色
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
    color_features = np.concatenate((hist_h, hist_s, hist_v))
    if color_features.sum() > 0:
        color_features /= color_features.sum()
    

    # --- Grid Local Features（局所特徴） ---
    grid_features = []

    h, w = image.shape[:2]

    grid_h = h // 3
    grid_w = w // 3

    for i in range(3):
        for j in range(3):

            cell = image[
                i * grid_h:(i + 1) * grid_h,
                j * grid_w:(j + 1) * grid_w
            ]

            # 空画像対策
            if cell.size == 0:
                continue

            # HSV変換
            hsv_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

            # Hヒストグラム
            hist_h = cv2.calcHist(
                [hsv_cell],
                [0],
                None,
                [16],
                [0, 180]
            ).flatten()

            # 正規化
            hist_h = hist_h.astype(np.float32)
            hist_h /= (hist_h.sum() + 1e-7)

            # Gray
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # エッジ量
            edges = cv2.Canny(gray_cell, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size

            # LBP
            lbp_cell = local_binary_pattern(
                gray_cell,
                P=8,
                R=1,
                method="uniform"
            )

            lbp_hist, _ = np.histogram(
                lbp_cell.ravel(),
                bins=16,
                range=(0, 16)
            )

            lbp_hist = lbp_hist.astype(np.float32)
            lbp_hist /= (lbp_hist.sum() + 1e-7)

            # 特徴追加
            grid_features.extend(hist_h)
            grid_features.extend(lbp_hist)
            grid_features.append(edge_density)

    grid_features = np.array(grid_features, dtype=np.float32)

    combined = np.concatenate((
        cnn_features * 3.0,
        color_features * 0.3,
        grid_features * 1.5
    ))

    return combined


# --- 特徴量の名前と範囲を定義 ---
def get_feature_info(image_size):
    cnn_dim = 512
    color_dim = 180 + 256 + 256
    grid_dim = (16 + 16 + 1) * 9

    feature_info = {
        'CNN': {
            'start': 0,
            'end': cnn_dim
        },

        'Color Histogram': {
            'start': cnn_dim,
            'end': cnn_dim + color_dim
        },

        'Grid Local Features': {
            'start': cnn_dim + color_dim,
            'end': cnn_dim + color_dim + grid_dim
        }
    }
    return feature_info


# --- データ読み込みと拡張関数 ---
def load_and_augment_data(data_dir, image_size):
    all_features = []
    all_labels = []
    all_filepaths = []
    all_image_ids = []
    image_id_counter = 0
    # クラス名取得
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f">> 分類カテゴリ: {class_names}")
    # 元画像枚数カウント
    total_original_images = sum([len(files) for r, d, files in os.walk(data_dir) if files])
    processed_original_images = 0
    # クラスごとに処理
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                continue
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: 画像を読み込めませんでした: {img_path}")
                    continue
                img = cv2.resize(img, image_size)
                relative = os.path.relpath(img_path, DATA_DIR)
                # データ拡張
                augmented_images = apply_data_augmentation(img)
                # 各拡張画像から特徴抽出
                for idx, aug_img in enumerate(augmented_images):

                    # cacheファイル名
                    feature_cache_name = os.path.join(
                        FEATURE_DIR,
                        relative.replace("\\", "_") + f"_aug{idx}.npy"
                    )

                    # cacheが存在するなら読み込み
                    if os.path.exists(feature_cache_name):

                        features = np.load(feature_cache_name)

                    else:
                        # 特徴抽出
                        features = extract_features(aug_img)

                        # 保存
                        if features is not None:
                            np.save(feature_cache_name, features)
                    
                    if features is not None:
                        all_features.append(features)
                        all_labels.append(class_idx)
                        all_filepaths.append(img_path)
                        all_image_ids.append(image_id_counter)
                        image_id_counter += 1
            except Exception as e:
                print(f"Warning: 画像 {img_path} の処理中にエラー: {e}")
                continue
            finally:
                processed_original_images += 1
                if processed_original_images % 100 == 0 or processed_original_images == total_original_images:
                    print(f">>   {processed_original_images}/{total_original_images} 個のオリジナル画像を処理中...")
    print(f">> 合計 {len(all_features)} 枚の拡張画像を読み込みました。")
    return np.array(all_features), np.array(all_labels), np.array(all_filepaths), np.array(all_image_ids), class_names

def extract_from_paths(paths, labels, image_size, augment=False):
        all_features = []
        all_labels = []
        all_filepaths = []
        all_image_ids = []

        image_id = 0

        for path, label in zip(paths, labels):

            img = cv2.imread(path)
            if img is None:
                continue

            relative = os.path.relpath(path, DATA_DIR)
            
            cache_name = os.path.join(
                CROP_DIR,
                relative
            )
            
            os.makedirs(
                os.path.dirname(cache_name),
                exist_ok=True
            )
            

            if os.path.exists(cache_name):
                img = cv2.imread(cache_name)
            else:
                img, _ = detect_and_crop_animal(img, yolo_model)
                if img is None:
                    continue
                cv2.imwrite(cache_name, img)

            img = cv2.resize(img, image_size)

            
            images = [img]

            if augment:
                images = apply_data_augmentation(img)

            for idx, im in enumerate(images):
                feature_cache_name = os.path.join(
                    FEATURE_DIR,
                    relative.replace("\\", "_") + f"_aug{idx}.npy"
                )

                if os.path.exists(feature_cache_name):
                    features = np.load(feature_cache_name)
                else:
                    features = extract_features(im)
                    np.save(feature_cache_name, features)



                if features is not None:
                    all_features.append(features)
                    all_labels.append(label)
                    all_filepaths.append(path)
                    all_image_ids.append(image_id)
                    image_id += 1

        return (
            np.array(all_features),
            np.array(all_labels),
            np.array(all_filepaths),
            np.array(all_image_ids)
        )


# --- データ準備関数 ---
def prepare_data(X_train, X_test, y_train, y_test):

    print("\n>> --- データ準備 ---")

    print(f"train: {len(X_train)}")
    print(f"test: {len(X_test)}")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("標準化完了")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# --- モデル訓練と評価関数 ---
def train_and_evaluate_model(X_train, X_test, y_train, y_test, class_names, scaler, model_name, random_state=RANDOM_STATE, feature_info=None):
    print(f"\n>> --- {model_name}モデル訓練と評価 ---")
    print("\n>> --- PCAによる次元削減 ---")
    pca = PCA(n_components=0.95, random_state=random_state) 
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    print(f">> 次元削減: {X_train.shape[1]} -> {X_train_reduced.shape[1]}")
    print("\n>> --- Random Forest ハイパーパラメータチューニング ---")
    
    idx = np.random.choice(
        len(X_train_reduced),
        min(2000, len(X_train_reduced)),
        replace=False
    )

    visualize_pca_space(
        X_train_reduced[idx],
        y_train[idx],
        class_names,
        model_name
    )

    param_dist = {  
        'n_estimators': [100, 200, 300],         # 決定木の本数
        'max_features': ['sqrt', 'log2', None],  # 分割時に使用する特徴量数
        'max_depth': [10, 20, None],             # 木の最大深さ
        'min_samples_split': [2, 5],             # ノード分割に必要な最小サンプル数
        'min_samples_leaf': [1, 2],              # 葉ノードに必要な最小サンプル数
        'bootstrap': [True, False]               # ブートストラップサンプリングを使うか
    }
    
    rf = RandomForestClassifier(random_state=random_state)
    
    search = RandomizedSearchCV(
        rf,                                     # RandomForestClassifier
        param_dist,                             # 試すパラメータ候補
        n_iter=10,                              # 試行回数
        cv=3,                                   # n分割交差検証
        verbose=1,                              # 少し学習中の進行状況を表示
        random_state=random_state,              # 乱数固定
        n_jobs=-1                               # 全CPUを使用
    )
    
    search.fit(X_train_reduced, y_train)
    best_rf = search.best_estimator_
    print(f">> 最適パラメータ: {search.best_params_}")
    
    y_pred = best_rf.predict(X_test_reduced)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f">> [{model_name}]Accuracy: {acc:.4f}")
    print(f">> [{model_name}]F1 Score: {f1:.4f}")
    print(f">> [{model_name}]Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'[{model_name}] Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # --- 特徴量全体の重要度を可視化 ---
    print("\n>> --- 特徴量全体の重要度 ---")
    if feature_info:
        # PCA前のデータを使ってモデルを再訓練し、特徴量の重要度を取得
        # ハイパーパラメータはチューニング結果を使用
        params = search.best_params_.copy()
        if 'random_state' in params:
            del params['random_state']

        rf_no_pca = RandomForestClassifier(random_state=random_state, **params)
        rf_no_pca.fit(X_train, y_train)
        
        importances = rf_no_pca.feature_importances_
        original_feature_importance = np.zeros(len(feature_info))
        feature_names = list(feature_info.keys())
        
        for i, (name, info) in enumerate(feature_info.items()):
            original_feature_importance[i] = np.sum(importances[info['start']:info['end']])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(feature_names, original_feature_importance)
        ax.set_title(f"{model_name} 全体の特徴量重要度")
        ax.set_xlabel("特徴量タイプ")
        ax.set_ylabel("重要度 (合計)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # --- モデルの保存 ---
    print("\n>> --- モデルをファイルに保存中 ---")

    joblib.dump(best_rf, f'{model_name}_model.joblib')
    joblib.dump(pca, f'{model_name}_pca.joblib')
    joblib.dump(scaler, f'{model_name}_scaler.joblib')

    joblib.dump(rf_no_pca, f"{model_name}_rf_no_pca.joblib")

    print("モデルが正常に保存されました。")


    return (
        best_rf,
        pca,
        scaler,    
        search.best_params_,
        rf_no_pca
    )

def visualize_pca_space(X_reduced, y, class_names, model_names):
    plt.figure(figsize=(12, 10))
    for i, cname in enumerate(class_names):
        plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], label=cname, alpha=0.6, s=20)
    plt.title(f'{model_names} 品種特徴のPCA分布')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.show()




# --- メイン実行ブロック ---
if __name__ == "__main__":
    start_time = time.perf_counter()
    
    print("--- 犬猫の品種分類・類似画像検索プログラム ---")
    
    # 特徴量グループの情報を取得
    feature_info = get_feature_info(IMAGE_SIZE)

    # 元画像のみ収集追加
    filepaths, labels, class_names = collect_image_paths(DATA_DIR)

    # YOLO確認用
    random_path = random.choice(filepaths)

    test_img = cv2.imread(random_path)

    cropped_img, detected_label = detect_and_crop_animal(test_img, yolo_model)

    print("選ばれた画像:", random_path)
    print("YOLO判定:", detected_label)

    # 表示
    plt.figure(figsize=(5,5))

    img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.title(f"YOLO Detection: {detected_label}")
    plt.axis("off")
    plt.show()

    # 元画像でtrain/test分割
    train_paths, test_paths, y_train_orig, y_test_orig = train_test_split(
        filepaths,
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=RANDOM_STATE
    )
    print("train original:", len(train_paths))
    print("test original:", len(test_paths))

    # 猫・犬専用ラベル作成
    cat_label_map = {
        name: idx
        for idx, name in enumerate(cat_class_names)
    }

    dog_label_map = {
        name: idx
        for idx, name in enumerate(dog_class_names)
    }


    # 猫と犬でデータを分離
    cat_train_paths = []
    cat_train_labels = []

    dog_train_paths = []
    dog_train_labels = []

    for path, label in zip(train_paths, y_train_orig):

        class_name = class_names[label]
        
        if class_name in cat_classes:

            cat_train_paths.append(path)

            cat_train_labels.append(
                cat_label_map[class_name]
            )

        elif class_name in dog_classes:

            dog_train_paths.append(path)

            dog_train_labels.append(
                dog_label_map[class_name]
            )

    cat_test_paths = []
    cat_test_labels = []

    dog_test_paths = []
    dog_test_labels = []

    for path, label in zip(test_paths, y_test_orig):

        class_name = class_names[label]

        if class_name in cat_classes:
            cat_test_paths.append(path)
            cat_test_labels.append(
                cat_label_map[class_name]
            )

        elif class_name in dog_classes:
            dog_test_paths.append(path)
            dog_test_labels.append(
                dog_label_map[class_name]
            )

    print("猫 train:", len(cat_train_paths))     # 学習画像の数
    print("犬 train:", len(dog_train_paths))


    # 猫モデル
    cat_features_train, cat_y_train, cat_filepaths_train, cat_image_ids_train = extract_from_paths(
        cat_train_paths,
        cat_train_labels,
        IMAGE_SIZE,
        augment=True
    )

    cat_features_test, cat_y_test, cat_filepaths_test, cat_image_ids_test = extract_from_paths(
        cat_test_paths,
        cat_test_labels,
        IMAGE_SIZE,
        augment=False
    )

    cat_X_train_scaled, cat_X_test_scaled, cat_y_train, cat_y_test, cat_scaler = prepare_data(
        cat_features_train,
        cat_features_test,
        cat_y_train,
        cat_y_test
    )

    cat_model, cat_pca, cat_scaler, cat_best_params, cat_rf_no_pca = train_and_evaluate_model(
        cat_X_train_scaled,
        cat_X_test_scaled,
        cat_y_train,
        cat_y_test,
        cat_class_names,
        cat_scaler,
        "猫",
        feature_info=feature_info
    )

    cat_X_train_reduced = cat_pca.transform(cat_X_train_scaled)

    joblib.dump(
        {
            "X_train_reduced": cat_X_train_reduced,
            "y_train": cat_y_train,
            "filepaths": cat_filepaths_train
        },
        "cat_database.joblib"
    )

    # 犬モデル
    dog_features_train, dog_y_train, dog_filepaths_train, dog_image_ids_train = extract_from_paths(
        dog_train_paths,
        dog_train_labels,
        IMAGE_SIZE,
        augment=True
    )

    dog_features_test, dog_y_test, dog_filepaths_test, dog_image_ids_test = extract_from_paths(
        dog_test_paths,
        dog_test_labels,
        IMAGE_SIZE,
        augment=False
    )

    dog_X_train_scaled, dog_X_test_scaled, dog_y_train, dog_y_test, dog_scaler = prepare_data(
        dog_features_train,
        dog_features_test,
        dog_y_train,
        dog_y_test
    )

    dog_model, dog_pca, dog_scaler, dog_best_params, dog_rf_no_pca = train_and_evaluate_model(
        dog_X_train_scaled,
        dog_X_test_scaled,
        dog_y_train,
        dog_y_test,
        dog_class_names,
        dog_scaler,
        "犬",
        feature_info=feature_info
    )

    dog_X_train_reduced = dog_pca.transform(dog_X_train_scaled)

    joblib.dump(
        {
            "X_train_reduced": dog_X_train_reduced,
            "y_train": dog_y_train,
            "filepaths": dog_filepaths_train
        },
        "dog_database.joblib"
    )
    
    print("学習完了")
    print("モデル読み込み完了")
    
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    print(f"\n総実行時間: {elapsed_time:.2f} 秒")
    print(f"総実行時間: {elapsed_time / 60:.2f} 分")

    print("\n--- プログラム終了 ---")
