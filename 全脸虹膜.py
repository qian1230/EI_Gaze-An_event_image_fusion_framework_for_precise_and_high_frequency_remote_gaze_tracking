import cv2
import numpy as np
import dlib

# 8-邻域像素点的坐标定义
connects = [
    (-1, -1), (0, -1), (1, -1), (1, 0),
    (1, 1), (0, 1), (-1, 1), (-1, 0)
]

# 读取图像
src = cv2.imread("450.jpg")  # 替换为你的全脸图片路径

# 转换为灰度图像
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 使用dlib检测人脸和特征点
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
dets = detector(src, 1)

# 检查是否检测到人脸
if dets:
    for i, face in enumerate(dets):
        x, y, w, h = dlib.rectangle.left(face), dlib.rectangle.top(face), dlib.rectangle.width(
            face), dlib.rectangle.height(face)

        # 提取出每一张脸的特征点
        face_feature = predictor(src, face)

        # 获取左眼的特征点坐标
        left_eye_points = [(face_feature.part(n).x, face_feature.part(n).y) for n in range(36, 42)]

        # 计算左眼的边界框
        left_eye_x = min(pt[0] for pt in left_eye_points)
        left_eye_y = min(pt[1] for pt in left_eye_points)
        left_eye_w = max(pt[0] for pt in left_eye_points) - left_eye_x
        left_eye_h = max(pt[1] for pt in left_eye_points) - left_eye_y

        # 截取左眼区域
        left_eye_gray = src_gray[left_eye_y:left_eye_y + left_eye_h, left_eye_x:left_eye_x + left_eye_w]

        # 初始化生长结果图像，大小与原图相同，数据类型为8位无符号整型
        grow_res = np.zeros_like(left_eye_gray, dtype=np.uint8)

        # 标记图像，用于记录哪些像素点已被访问
        flagMat = grow_res.copy()

        # 将灰度图像转换为二值图像
        _, binary_eye = cv2.threshold(left_eye_gray, 41, 255, cv2.THRESH_BINARY)

        # 设置虹膜中心点作为种子点
        center_x = int(left_eye_w / 2)
        center_y = int(left_eye_h / 2)
        seeds = [(center_x, center_y)]
        seeds_stack = []
        for seed in seeds:
            seeds_stack.append(seed)

        # 区域生长算法
        while seeds_stack:
            seed = seeds_stack.pop()
            x, y = seed
            flagMat[y, x] = 1  # 标记为已访问
            for i in range(8):
                tempx, tempy = x + connects[i][0], y + connects[i][1]
                if 0 <= tempx < left_eye_w and 0 <= tempy < left_eye_h:
                    if binary_eye[tempy, tempx] == 255 and flagMat[tempy, tempx] == 0:
                        grow_res[tempy, tempx] = 255  # 生长
                        flagMat[tempy, tempx] = 1  # 标记
                        seeds_stack.append((tempx, tempy))  # 添加为新种子

        # 使用Canny算法进行边缘检测
        canny_dst = cv2.Canny(grow_res, 100, 200)

        # 查找轮廓
        contours, hierarchy = cv2.findContours(canny_dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 初始化最大圆的中心和半径
        max_center = None
        max_radius = 0

        # 遍历所有轮廓，寻找最小外接圆
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            if radius > max_radius:
                max_radius = radius
                max_center = center

        # 将局部坐标转换为全局坐标
        if max_center is not None:
            global_center = (max_center[0] + left_eye_x, max_center[1] + left_eye_y)
            cv2.circle(src, global_center, int(max_radius), (0, 255, 0), 2)

        # 显示结果
        cv2.namedWindow("Iris Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Iris Detection", src)
        cv2.imwrite("iris_detected.jpg", src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("No face detected in the image.")