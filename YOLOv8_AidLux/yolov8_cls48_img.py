# aidlux相关
from cvs import *
import aidlite_gpu
from utils import detect_postprocess, preprocess_img, draw_detect_res, scale_boxes

import time
# import requests
import cv2


# tflite模型
model_path = '/home/code/YOLOv8_AidLux/models/best_float32.tflite'
# 定义输入输出shape
in_shape = [1 * 640 * 640 * 3 * 4]  # HWC, float32
out_shape = [1 * 8400 * 52 * 4]  # 8400: total cells, 52 = 48(num_classes) + 4(xywh), float32

# AidLite初始化
aidlite = aidlite_gpu.aidlite()
# 载入模型
res = aidlite.ANNModel(model_path, in_shape, out_shape, 4, 0)
print(res)

''' 读取本地图片 '''
image_path = "/home/code/YOLOv8_AidLux/samples/000000.jpg"
image = cv2.imread(image_path)
''' 读取本地视频 '''
# cap = cvs.VideoCapture('/codes/Face_Rec/test/videos/Kuangbiao2.mp4')
''' 读取手机后置摄像头 '''
# cap = cvs.VideoCapture(0)
frame = image
time0 = time.time()
sz = frame.shape
print(sz)
# 预处理
img = preprocess_img(frame, target_shape=(640, 640), div_num=255, means=None, stds=None)
frame_id = 0
aidlite.setInput_Float32(img, 640, 640)
# 推理
aidlite.invoke()
preds = aidlite.getOutput_Float32(0)
preds = preds.reshape(1, 52, 8400)
preds = detect_postprocess(preds, frame.shape, [640, 640, 3], conf_thres=0.25, iou_thres=0.45)
print('1 batch takes {} s'.format(time.time() - time0))
if len(preds) != 0:
        preds[:, :4] = scale_boxes([640, 640], preds[:, :4], frame.shape)
        frame = draw_detect_res(frame, preds)
cvs.imshow(frame)
cv2.imwrite("./out/frame_best_float32.jpg",frame)
# img_processed = cv2.resize(frame, (640, 640))
