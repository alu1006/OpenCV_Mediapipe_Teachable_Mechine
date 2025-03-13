import cv2
import mediapipe as mp

# 初始化 MediaPipe 人臉偵測
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 載入圖片
image_path = 'Image Model/test.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 使用 MediaPipe 偵測臉部
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(image_rgb)

    # 若偵測到臉部，則繪製框線
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

# 顯示結果
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 儲存偵測後圖片
cv2.imwrite('face_detected.jpg', image)
