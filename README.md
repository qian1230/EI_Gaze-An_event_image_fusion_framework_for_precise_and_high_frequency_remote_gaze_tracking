# event-base-pipeline-gaze-tracking
请自行下载shape_predictor_68_face_landmarks.dat
本代码为结合事件的非端到端特征提取高频远程注视追踪研究
Remote gaze tracking has emerged as a pivotal technology across multiple domains, in-
cluding web interface optimization, driver drowsiness prevention, and immersive vir-
tual reality systems. However, conventional image-based gaze tracking methods are
inherently limited by low temporal resolution (50Hz) and measurement constraints as-
sociated with distant capture, which often compromise their accuracy. To address these
challenges, we propose EI-Gaze, a novel precise and high-frequency remote gaze track-
ing framework specifically designed for scenarios where near-eye wearable devices are
impractical. EI-Gaze integrates event cameras to capture the rapid movements of the
iris with high temporal resolution and RGB cameras to obtain clear facial and head
images. This hybrid setup enables the extraction of crucial gaze-related features, in-
cluding facial center, head pose, and iris center, through a series of designed compo-
nents. Leveraging a ridge regression model, EI-Gaze can accurately predict the user’s
gaze direction. Comprehensive evaluations on RGBE-Gaze, a large-scale dataset for
hybrid event and RGB camera-based gaze tracking, demonstrate that EI-Gaze achieves
a remarkable temporal resolution of 250Hz. Compared with two state-of-the-art meth-
ods, RGBE-Gaze and MnistNet, EI-Gaze reduces the mean angular error by 13.3% and
27.7%, respectively, with an error of only 3.840°. Lastly, EI-Gaze provides a reliable
solution for remote gaze tracking at 250Hz, enabling practical applications such as real-
time driving fatigue monitoring and athletic training optimization.

应用的数据主要为 RGBE-GAZE 数据集 

![image](https://github.com/user-attachments/assets/6d64ba22-5189-48fd-abae-4d6465d7b35c)
![image](https://github.com/
qian1230/EI_Gaze-An_event_image_fusion_framework_for_precise_and_high_
frequency_remote_gaze_tracking/6312d69f65d183c0171be79f32b627c.png).
