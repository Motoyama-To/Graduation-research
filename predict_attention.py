# 猫+犬 品種分類・類似画像検索プログラム

# モデル・データ読み込み -- YOLO切り抜き -- model.eval() -- softmax 
# -- 類似画像検索 -- Wikipedia説明 

# predict_attention.py


model = BreedClassifier(num_classes=10)

model.load_state_dict(
    torch.load("cat_model.pth", map_location=device)
)

model.eval()