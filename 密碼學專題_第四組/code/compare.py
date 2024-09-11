import cv2
import matplotlib.pyplot as plt
import numpy as np
from functions import *

original_img = cv2.imread('input.jpg')
img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
gray = cvtGray(img)

imgRcv = cv2.imread("encoded_image.jpg")
imgRcv = cv2.cvtColor(imgRcv, cv2.COLOR_BGR2RGB)
grayRcv = cvtGray(imgRcv)

# 嵌入前後灰階對比圖
plt.figure(figsize=(12, 6)), plt.suptitle('Grayscale')
plt.subplot(1, 2, 1), plt.title('Origin')
plt.hist(gray.ravel(), 256)
plt.show(block=False)

# 驗證灰階不變性
plt.subplot(1, 2, 2), plt.title('Modified')
plt.hist(grayRcv.ravel(), 256)
plt.show(block=False)
print(f'Ensure the grayscale invariant: {np.all(gray == grayRcv)}')

plt.savefig('Grayscale.jpg')
plt.show()