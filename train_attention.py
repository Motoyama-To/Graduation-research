# 猫+犬 品種分類・類似画像検索プログラム

# --- モデル全体の処理 ---
# 入力画像
#   ↓ load_input_image()
# YOLOv8で猫・犬を検出
#   ↓ detect_and_crop_animal()
# 動物部分を切り出し
#   ↓ crop_animal_region()
# ResNet18で特徴抽出
#   ↓ extract_features_by_resnet18()
# CBAMで重要部分を強調
#   ├─ CAM：重要な特徴チャネルを判断
#   └─ SAM：重要な位置を判断
#   ↓ apply_cbam_attention()
# Adaptive Average Poolingで特徴を圧縮
#   ↓ pool_features()
# Linear層で分類
#   ↓ classify_breed()
# 品種分類結果（猫10種・犬10種）


# YOLO前処理 + ResNet18 + CBAM による End-to-End分類器

# train-attention.py

import os
import cv2
import numpy as np
import random
from collections import Counter
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
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

# --- 画像前処理とデータ拡張のためのTransform ---
train_transform = transforms.Compose([

    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),   # 50%で左右反転
    transforms.RandomRotation(10),       # ±10度以内でランダム回転
    transforms.ColorJitter(              # 明るさ・コントラストをランダム変更
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(                # 正規化
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# --- テスト用のTransform ---
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
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



# --- 特徴注意モジュール Channel Attention Module  どの特徴を見るか ---
class CAM(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 全結合層の代わりの1×1畳み込み
        self.fc = nn.Sequential(       
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),   # チャネル数を圧縮
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)    # 元のチャネル数に戻す
        )

        self.sigmoid = nn.Sigmoid()             # 0～1の重みへ変換

    def forward(self, x):

        avg_out = self.fc(self.avg_pool(x))     # 注意重み計算
        max_out = self.fc(self.max_pool(x))

        return self.sigmoid(avg_out + max_out)

# --- 空間注意モジュール Spatial Attention Module  どこを見るか ---
class SAM(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(                  # 7×7畳み込みで空間方向の注意を計算
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(               # チャネル方向の平均
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        x = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        x = self.conv(x)

        return self.sigmoid(x)

# --- CBAMモジュール ---
class CBAM(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.ca = CAM(channels)
        self.sa = SAM()

    def forward(self, x):

        x = self.ca(x) * x              # 重要な特徴チャネルを強調
        x = self.sa(x) * x              # 重要な空間領域を強調

        return x

# --- モデル学習関数 ---
def calculate_loss_and_accuracy(model, data_loader, criterion):
    model.eval()
    running_loss = 0                    # 損失の合計を保存
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs, 1)

            running_loss += loss.item()
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    if len(data_loader) == 0 or total == 0:
        return 0, 0

    return running_loss / len(data_loader), correct / total


# --- モデルを学習する関数 ---
def train_model(
    model,
    train_loader,
    val_loader=None,
    num_epochs=10,
    title="Model"
):

    loss_history = []
    accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    best_val_accuracy = 0
    best_epoch = 0

    criterion = nn.CrossEntropyLoss()           # 損失関数（多クラス分類）

    optimizer = torch.optim.Adam(               # Adam最適化
        model.parameters(),
        lr=0.0001
    )

    model.to(device)

    if len(train_loader) == 0:
        raise ValueError("No training data.")

    for epoch in range(num_epochs):             # 学習
        epoch_start_time = time.perf_counter()
        model.train()
        running_loss = 0

        for images, labels in train_loader:     # ミニバッチ学習

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()                     # 誤差逆伝播       
            optimizer.step()                    # パラメータ更新
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)         # 損失累積
        loss_history.append(epoch_loss)

        _, train_accuracy = calculate_loss_and_accuracy(
            model,
            train_loader,
            criterion                           # 損失関数
        )
        accuracy_history.append(train_accuracy)

        # 検証用データがある場合
        if val_loader is not None:
            val_loss, val_accuracy = calculate_loss_and_accuracy(
                model,
                val_loader,
                criterion
            )
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1

        epoch_time = time.perf_counter() - epoch_start_time

        if val_loader is not None:
            # 学習損失・学習正解率・検証損失・検証正解率・時間
            print(
                f"{title} Epoch {epoch+1}/{num_epochs} | "
                f"train_loss={epoch_loss:.4f} | "
                f"train_acc={train_accuracy:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_accuracy:.4f} | "
                f"time={epoch_time:.2f}s"
            )
        else:
            print(
                f"{title} Epoch {epoch+1}/{num_epochs} | "
                f"train_loss={epoch_loss:.4f} | "
                f"train_acc={train_accuracy:.4f} | "
                f"time={epoch_time:.2f}s"
            )

    # 学習正解率の推移グラフ
    plt.figure()
    plt.plot(accuracy_history, label="train accuracy")
    if val_accuracy_history:
        plt.plot(val_accuracy_history, label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    # 学習損失の推移グラフ
    plt.figure()
    plt.plot(loss_history, label="train loss")
    if val_loss_history:
        plt.plot(val_loss_history, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss")
    plt.legend()
    plt.grid()
    plt.show()

    if val_loader is not None:
        print(f"{title} best val_acc={best_val_accuracy:.4f} at epoch {best_epoch}")

    return model, loss_history, accuracy_history, val_loss_history, val_accuracy_history


def evaluate_model(model, data_loader, class_names, title):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)       # 出力を確率（Softmax）へ変換
            _, pred = torch.max(outputs, 1)
            top3 = torch.topk(probabilities, k=min(3, len(class_names)), dim=1).indices.cpu()
            y_true.extend(labels.numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(probabilities.max(dim=1).values.cpu().numpy())

            labels_cpu = labels.cpu()

            # Top-3以内に正解ラベルが含まれている数を加算
            top3_correct += sum(
                labels_cpu[i].item() in top3[i].tolist()
                for i in range(labels_cpu.size(0))
            )
            total += labels_cpu.size(0)

    if total == 0:
        raise ValueError(f"No evaluation data for {title}.")

    print(f"\n--- {title} evaluation ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Top-3 Accuracy: {top3_correct / total:.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Average confidence: {np.mean(y_prob):.4f}")                 # 平均予測確率
    print(classification_report(
        y_true, 
        y_pred, 
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    ))

    print(f"\n--- {title} per-class accuracy ---")
    for class_idx, class_name in enumerate(class_names):
        class_total = sum(1 for label in y_true if label == class_idx)
        class_correct = sum(
            1
            for true_label, pred_label in zip(y_true, y_pred)
            if true_label == class_idx and pred_label == class_idx
        )
        class_accuracy = class_correct / class_total if class_total else 0
        
        print(f"{class_name}: {class_accuracy:.4f} ({class_correct}/{class_total})")

    cm = confusion_matrix(y_true, y_pred)                               # 混同行列を作成
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"{title} confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    cm_normalized = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1
    )
    plt.title(f"{title} normalized confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# --- 品種分類モデル ---
class BreedClassifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.features = nn.Sequential(              # 最終分類層を除いた特徴抽出部分
            *list(self.backbone.children())[:-2]
        )

        self.cbam = CBAM(512)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(
            512,
            num_classes
        )

    def forward(self, x):

        x = self.features(x)
        x = self.cbam(x)                # CBAMで強調
        x = self.pool(x)
        x = torch.flatten(x, 1)         # ベクトル化
        x = self.fc(x)                  # 品種分類
        return x

# --- データセットクラス ---
class PetDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        transform=None
    ):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = load_or_create_crop(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # BGR → RGBへ変換

        if self.transform:
            img = self.transform(img)

        return img, label


def get_crop_cache_path(img_path):
    relative = os.path.relpath(img_path, DATA_DIR)
    return os.path.join(CROP_DIR, relative)


# --- cache画像を読み込むor新しく作成する関数 ---
def load_or_create_crop(img_path, return_label=False):
    cache_name = get_crop_cache_path(img_path)

    if os.path.exists(cache_name):
        img = cv2.imread(cache_name)
        if img is not None:
            if return_label:
                return img, "cached"
            return img

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    cropped_img, detected_label = detect_and_crop_animal(img, yolo_model)
    if cropped_img is None:
        cropped_img = img

    os.makedirs(os.path.dirname(cache_name), exist_ok=True)
    cv2.imwrite(cache_name, cropped_img)
    if return_label:
        return cropped_img, detected_label
    return cropped_img


# --- 全画像のYOLO切り抜きcacheを事前作成する関数 ---
def prepare_crop_cache(paths):
    crop_stats = Counter()

    for i, path in enumerate(paths, start=1):
        _, detected_label = load_or_create_crop(path, return_label=True)
        crop_stats[detected_label] += 1
        if i % 100 == 0 or i == len(paths):
            print(f">> YOLO crop cache: {i}/{len(paths)}")

    print(
        "YOLO crop summary:",
        f"cat={crop_stats['cat']}",
        f"dog={crop_stats['dog']}",
        f"unknown={crop_stats['unknown']}",
        f"cached={crop_stats['cached']}"
    )

    return crop_stats

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


def print_class_counts(title, labels, class_names):
    counts = Counter(labels)
    print(f"\n--- {title} class counts ---")
    for class_idx, class_name in enumerate(class_names):
        print(f"{class_name}: {counts[class_idx]}")


# --- Main ---
if __name__ == "__main__":
    start_time = time.perf_counter()
    
    print("--- 犬猫の品種分類・類似画像検索プログラム ---")

    # 元画像のみ収集追加
    filepaths, labels, class_names = collect_image_paths(DATA_DIR)
    print("device:", device)
    print("total images:", len(filepaths))
    print_class_counts("All", labels, class_names)

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
    print_class_counts("Train original", y_train_orig, class_names)
    print_class_counts("Test original", y_test_orig, class_names)

    # 猫・犬専用ラベル作成
    cat_label_map = {
        name: idx
        for idx, name in enumerate(cat_class_names)
    }

    dog_label_map = {
        name: idx
        for idx, name in enumerate(dog_class_names)
    }

    # --- 猫と犬で学習データを分離する ---
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

    # --- 猫と犬でテストデータを分離する ---
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


    # --- 猫データを学習用(fit)と検証用(validation)に分割する ---
    cat_fit_paths, cat_val_paths, cat_fit_labels, cat_val_labels = train_test_split(
        cat_train_paths,
        cat_train_labels,
        test_size=0.2,
        stratify=cat_train_labels,
        random_state=RANDOM_STATE
    )

    # --- 犬データを学習用(fit)と検証用(validation)に分割する ---
    dog_fit_paths, dog_val_paths, dog_fit_labels, dog_val_labels = train_test_split(
        dog_train_paths,
        dog_train_labels,
        test_size=0.2,
        stratify=dog_train_labels,
        random_state=RANDOM_STATE
    )

    print("cat fit:", len(cat_fit_paths))
    print("cat val:", len(cat_val_paths))
    print("cat test:", len(cat_test_paths))
    print("dog fit:", len(dog_fit_paths))
    print("dog val:", len(dog_val_paths))
    print("dog test:", len(dog_test_paths))
    print_class_counts("Cat fit", cat_fit_labels, cat_class_names)              # 猫の学習用データのクラス数
    print_class_counts("Cat val", cat_val_labels, cat_class_names)
    print_class_counts("Cat test", cat_test_labels, cat_class_names)
    print_class_counts("Dog fit", dog_fit_labels, dog_class_names)
    print_class_counts("Dog val", dog_val_labels, dog_class_names)
    print_class_counts("Dog test", dog_test_labels, dog_class_names)

    prepare_crop_cache(list(train_paths) + list(test_paths))

    # -- 猫データセットとデータローダーの作成 --
    cat_train_dataset = PetDataset(
        cat_fit_paths,
        cat_fit_labels,
        train_transform
    )

    cat_val_dataset = PetDataset(
        cat_val_paths,
        cat_val_labels,
        test_transform
    )

    cat_test_dataset = PetDataset(
        cat_test_paths,
        cat_test_labels,
        test_transform
    )

    cat_train_loader = DataLoader(
        cat_train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    cat_test_loader = DataLoader(
        cat_test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    cat_val_loader = DataLoader(
        cat_val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # 猫学習とモデル保存
    cat_model = BreedClassifier(
        num_classes=10
    )

    cat_model, cat_loss, cat_accuracy, cat_val_loss, cat_val_accuracy = train_model(
        cat_model,
        cat_train_loader,
        val_loader=cat_val_loader,
        num_epochs=10,
        title="Cat"
    )

    evaluate_model(
        cat_model,
        cat_test_loader,
        cat_class_names,
        "Cat"
    )

    torch.save(
        cat_model.state_dict(),
        "cat_model.pth"
    )

    plt.figure()
    plt.plot(cat_loss, label="train loss")
    plt.plot(cat_val_loss, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cat Loss")
    plt.legend()
    plt.grid()
    plt.show()


    # -- 犬データセットとデータローダーの作成 --
    dog_train_dataset = PetDataset(
        dog_fit_paths,
        dog_fit_labels,
        train_transform
    )

    dog_val_dataset = PetDataset(
        dog_val_paths,
        dog_val_labels,
        test_transform
    )

    dog_test_dataset = PetDataset(
        dog_test_paths,
        dog_test_labels,
        test_transform
    )

    dog_train_loader = DataLoader(
        dog_train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    dog_test_loader = DataLoader(
        dog_test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    dog_val_loader = DataLoader(
        dog_val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # 犬学習とモデル保存
    dog_model = BreedClassifier(
        num_classes=10
    )

    dog_model, dog_loss, dog_accuracy, dog_val_loss, dog_val_accuracy = train_model(
        dog_model,
        dog_train_loader,
        val_loader=dog_val_loader,
        num_epochs=10,
        title="Dog"
    )

    evaluate_model(
        dog_model,
        dog_test_loader,
        dog_class_names,
        "Dog"
    )

    torch.save(
        dog_model.state_dict(),
        "dog_model.pth"
    )

    plt.figure()
    plt.plot(dog_loss, label="train loss")
    plt.plot(dog_val_loss, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Dog Loss")
    plt.legend()
    plt.grid()
    plt.show()
    
    print("学習完了")
    print("モデル読み込み完了")
    
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    print(f"\n総実行時間: {elapsed_time:.2f} 秒")
    print(f"総実行時間: {elapsed_time / 60:.2f} 分")

    print("\n--- プログラム終了 ---")
