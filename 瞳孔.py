import cv2
import numpy as np

def detect_and_draw(image, cascade):
    A1 = 0
    A2 = 0
    B1 = 0
    B2 = 0
    faces = []
    colors = [
        (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0),
        (255, 128, 0), (255, 255, 0), (255, 0, 0), (255, 0, 255)
    ]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    t = cv2.getTickCount()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    t = (cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000
    print(f"detection time = {t:.2f} ms")
    i=0
    for (x, y, w, h), color in zip(faces, [colors[i % 8] for i in range(len(faces))]):
        center = (int(x + w / 2), int(y + h / 2))
        if i==0:
            A1=int(x + w / 2)
            A2=int(y + h / 2)
            i+=1
        else:
            B1=int(x + w / 2)
            B2=int(y + h / 2)
        radius = 2  # You can adjust this radius as needed
        cv2.circle(image, center, radius, color, 3)

    return A1,A2,B1,B2
import cv2
import numpy as np
import dlib
import pandas as pd
import os
if __name__ == "__main__":
    cascade_name = "haarcascade_eye_tree_eyeglasses.xml"
    cascade = cv2.CascadeClassifier(cascade_name)

    if cascade.empty():
        print(f"Error loading cascade file {cascade_name}")
        exit()

    image_folder = "./事件/"  # 替换为你的图片文件夹路径

    # 获取匹配的文件列表
    files = os.listdir(image_folder)
    # 筛选出以0-到99-开头且为JPG图片的文件
    jpg_files = []
    for file in files:
        if file.startswith(('0-', '1-', '2-', '3-', '4-', '5-', '6-', '7-', '8-', '9-')) and file.lower().endswith(
                ('.jpg', '.jpeg')):
            jpg_files.append(file)
        if len(jpg_files) == 100:  # 如果已经找到100张图片，就停止搜索
            break
    for i in range(10, 100):  # 从0到99
        prefix = f"{i}-"  # 格式化为两位数，例如"00-", "01-", ..., "99-"
        for file in files:
            if file.startswith(prefix) and file.lower().endswith(('.jpg', '.jpeg')):
                jpg_files.append(file)
                if len(jpg_files) == 100:  # 如果已经找到100张图片，就停止搜索
                    break
        if len(jpg_files) == 2:  # 如果已经找到100张图片，就停止搜索
            break

    # 打印找到的图片文件列表
    # print(jpg_files)
    # print(files)
    gaze_data = pd.read_csv('gazepoint_averages_updated.csv')
    i = 0
    # A = gaze_data['filename'].iloc[i]
    # print(A)
    # gaze_data['pu_left_eye_x'] = None
    # gaze_data['pu_left_eye_y'] = None
    # gaze_data['pu_right_eye_x'] = None
    # gaze_data['pu_right_eye_y'] = None
    for img_file in jpg_files:

        img_path = os.path.join(image_folder, img_file)
    img = cv2.imread('./事件/450.jpg')
    image = img

    A1, A2, B1, B2 = detect_and_draw(image, cascade)
    print(A1,A2,B1,B2)


#
#
# cv2.imshow("result", result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#plt.savefig('scatter_plot.png')  # 这将保存为 PNG 格式的图片
