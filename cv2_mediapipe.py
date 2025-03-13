from keras.models import load_model
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import cv2
import mediapipe as mp

# Keras模型載入
model = load_model("Image Model/keras_Model.h5", compile=False)
class_names = open("Image Model/labels.txt", "r", encoding="utf-8").readlines()

# 圖片路徑
image_path = "Image Model/test.jpg"

# 圖片前處理 (for Keras)
image_pil = Image.open(image_path).convert("RGB")# 修正這裡，確保圖片是RGB模式，避免alpha channel問題
image_resized = ImageOps.fit(image_pil, (224, 224), Image.Resampling.LANCZOS)# 修正這裡，使用LANCZOS模式縮放圖片，避免縮放失真扭曲
image_array = np.asarray(image_resized)# 修正這裡，將PIL Image轉換成NumPy Array
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1# 修正這裡，將像素值調整到-1~1之間
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)#(batch_size, height, width, channels)
#此處設定的 batch_size = 1，表示模型每次只處理一張圖片進行預測。
data[0] = normalized_image_array

# Keras 預測
prediction = model.predict(data)
index = np.argmax(prediction)#可找出預測陣列中數值最大的索引值，也就是最可能的類別。
class_name = class_names[index].strip()[2:]
confidence_score = prediction[0][index]

# MediaPipe臉部偵測
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 偵測臉部並繪製框線
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(image_rgb)
    if results.detections:
        for detection in results.detections:
            print(detection)
            # mp.solutions.drawing_utils.draw_detection(image_cv, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
            
            # 用OpenCV畫矩形框 (藍色框，粗細為2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# OpenCV轉PIL加中文
image_final = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# 設定中文字體
font_path = "Image Model/msjh.ttc"  # 記得換成你的中文字型路徑
font = ImageFont.truetype(font_path, 32)

draw = ImageDraw.Draw(image_final)
label_text = f"{class_name}：{confidence_score:.2f}"

# 修正這裡，計算文字寬高
text_bbox = draw.textbbox((0, 0), label_text, font=font)
text_width = text_bbox[2] - text_bbox[0]
text_height = text_bbox[3] - text_bbox[1]

# 設定文字位置 (左下角避免遮擋)
margin = 10
draw.text((margin, image_final.height - text_height - margin),
          label_text, font=font, fill=(255, 255, 255))

# 顯示圖片
image_final.show()

# 儲存結果圖片
image_final.save("result_combined_chinese_fixed.jpg")
