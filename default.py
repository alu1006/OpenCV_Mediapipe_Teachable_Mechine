import cv2
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
# hack：修正 keras_model.h5 內的模型設定
import h5py
f = h5py.File("Image Model/keras_model.h5", mode="r+")
model_config_string = f.attrs.get("model_config")
if model_config_string.find('"groups": 1,') != -1:
    model_config_string = model_config_string.replace('"groups": 1,', '')
    f.attrs.modify('model_config', model_config_string)
    f.flush()
    model_config_string = f.attrs.get("model_config")
    assert model_config_string.find('"groups": 1,') == -1
f.close()
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("Image Model/keras_Model.h5", compile=False)

# Load the labels
class_names = open("Image Model/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("Image Model/test.jpg").convert("RGB")

# # resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# # turn the image into a numpy array
image_array = np.asarray(image)
'''
以上可以用OpenCV取代
# 使用OpenCV載入圖片
image_path = "Image Model/test.jpg"
image = cv2.imread(image_path)

# 調整圖片尺寸到 224x224（不裁切）
image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

# OpenCV預設BGR模式，轉成RGB
image_array = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

'''



# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
