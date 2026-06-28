# 猫+犬 品種分類・類似画像検索プログラム

# 入力画像 -- YOLOv8 検出と切り出し -- ResNet18特徴抽出 
# CBAM Attention
#   ├─ CAM : どの特徴を見るか
#   └─ SAM : どの位置を見るか
# Adaptive Average Pooling -- 全結合層 (Linear) -- 品種分類（猫10種・犬10種）


# YOLO前処理 + ResNet18 + CBAM による End-to-End分類器

# train-attention.py

import os
import cv2
import numpy as np
import random
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
def train_model(
    model,
    train_loader,
    num_epochs=10
):

    loss_history = []
    accuracy_history = []

    criterion = nn.CrossEntropyLoss()           # 損失関数（多クラス分類）

    optimizer = torch.optim.Adam(               # Adam最適化
        model.parameters(),
        lr=0.0001
    )

    model.to(device)

    for epoch in range(num_epochs):             # 学習
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
        if len(train_loader)==0:
            raise ValueError("No training data.")
        loss_history.append(epoch_loss)

        correct = 0
        total = 0

        model.eval()

        with torch.no_grad():

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, pred = torch.max(outputs,1)

                total += labels.size(0)
                correct += (pred==labels).sum().item()

        accuracy = correct/total

        accuracy_history.append(accuracy)

        model.train()

        print(
            f"Epoch {epoch+1}/{num_epochs}",
            f"Loss={epoch_loss:.4f}"
        )

    plt.figure()
    plt.plot(cat_accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(dog_accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()

    return model, loss_history, accuracy_history


def evaluate_model(model, data_loader, class_names, title):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(pred.cpu().numpy())

    print(f"\n--- {title} evaluation ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred, average='macro'))
    print(classification_report(
        y_true, 
        y_pred, 
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred)
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


def load_or_create_crop(img_path):
    cache_name = get_crop_cache_path(img_path)

    if os.path.exists(cache_name):
        img = cv2.imread(cache_name)
        if img is not None:
            return img

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    cropped_img, _ = detect_and_crop_animal(img, yolo_model)
    if cropped_img is None:
        cropped_img = img

    os.makedirs(os.path.dirname(cache_name), exist_ok=True)
    cv2.imwrite(cache_name, cropped_img)
    return cropped_img


def prepare_crop_cache(paths):
    for i, path in enumerate(paths, start=1):
        load_or_create_crop(path)
        if i % 100 == 0 or i == len(paths):
            print(f">> YOLO crop cache: {i}/{len(paths)}")

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


# --- Main ---
if __name__ == "__main__":
    start_time = time.perf_counter()
    
    print("--- 犬猫の品種分類・類似画像検索プログラム ---")

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

    prepare_crop_cache(list(train_paths) + list(test_paths))

    # -- 猫データセットとデータローダーの作成 --
    cat_train_dataset = PetDataset(
        cat_train_paths,
        cat_train_labels,
        train_transform
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

    # 猫学習とモデル保存
    cat_model = BreedClassifier(
        num_classes=10
    )

    cat_model, cat_loss, cat_accuracy = train_model(
        cat_model,
        cat_train_loader,
        num_epochs=10
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
    plt.plot(cat_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cat Loss")
    plt.grid()
    plt.show()


    # -- 犬データセットとデータローダーの作成 --
    dog_train_dataset = PetDataset(
        dog_train_paths,
        dog_train_labels,
        train_transform
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

    # 犬学習とモデル保存
    dog_model = BreedClassifier(
        num_classes=10
    )

    dog_model, dog_loss, dog_accuracy = train_model(
        dog_model,
        dog_train_loader,
        num_epochs=10
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
    plt.plot(dog_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Dog Loss")
    plt.grid()
    plt.show()
    
    print("学習完了")
    print("モデル読み込み完了")
    
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    print(f"\n総実行時間: {elapsed_time:.2f} 秒")
    print(f"総実行時間: {elapsed_time / 60:.2f} 分")

    print("\n--- プログラム終了 ---")
