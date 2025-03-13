# 圖片分類與臉部偵測應用 (Keras + MediaPipe)

本專案利用 **Keras 深度學習模型** 與 **MediaPipe 人臉偵測** 實現圖片分類和人臉定位，並於圖片上顯示中文分類標籤及信心分數。

---

## 🚀 功能介紹

- 使用 **Keras 預訓練模型** 進行圖片分類，並給出信心分數。
- 使用 **MediaPipe** 偵測圖片中的臉部位置，並以矩形框標示。
- 支援中文標籤顯示。

---

## 📂 專案檔案結構

```
Image Model/
├── keras_Model.h5
├── labels.txt
├── msjh.ttc (中文字體檔)
└── test.jpg (待預測圖片)
```

---

## 🛠️ 必要套件

```bash
pip install keras tensorflow opencv-python mediapipe pillow numpy h5py
```

---

## 📌 程式碼說明

### 1. 模型載入與圖片前處理
- 使用 PIL 調整圖片至模型所需大小 (224x224)，並進行正規化。

### 2. 圖片分類 (Keras)
- 使用 Keras 模型預測圖片類別，取得分類名稱與信心分數。

### 3. 人臉偵測 (MediaPipe)
- 利用 MediaPipe 定位圖片中人臉位置，並使用 OpenCV 畫出方框。

### 4. 顯示中文標籤
- 由於 OpenCV 不支援中文，需透過 PIL 在圖片上加入中文標籤（包含分類名稱與信心分數）。

### 5. 輸出結果
- 將結果圖片顯示並儲存為 `result_combined_chinese_fixed.jpg`。

---

## 📖 執行方法

```bash
python your_script.py
```

執行完畢後，結果圖片將會自動顯示並儲存在專案根目錄中。

---

## ⚠️ 注意事項

- 請確認 `msjh.ttc` 字體檔案存在於指定路徑 (`Image Model/msjh.ttc`)。
- 圖片路徑及模型檔案路徑可視需求進行修改。

---

## 🎯 應用情境

- 人臉識別與分類應用
- 智慧監控與自動標記系統
- 圖片分類與內容分析應用

