import cv2

# 加载预训练的Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图像
image_path = './other/2740-4854192064808.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path)

# 将图像转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测面部
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 遍历检测到的面部
for (x, y, w, h) in faces:
    # 计算面部的中心点
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    print(face_center_x, face_center_y)
    # 在图像上绘制矩形框和中心点
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.circle(image, (face_center_x, face_center_y), 5, (0, 255, 0), -1)

# 显示结果图像
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()