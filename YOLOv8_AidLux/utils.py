import cv2
import numpy as np
from numpy import array

# coco_class = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#         'hair drier', 'toothbrush']

coco_class = ['Cls0', 'Cls1', 'Cls2', 'Cls3', 'Cls4', 'Cls5', 'Cls6', 'Cls7', 'Cls8', 'Cls9',
        'Cls10', 'Cls11', 'Cls12', 'Cls13', 'Cls14', 'Cls15', 'Cls16', 'Cls17', 'Cls18', 'Cls19',
        'Cls20', 'Cls21', 'Cls22', 'Cls23', 'Cls24', 'Cls25', 'Cls26', 'Cls27', 'Cls28', 'Cls29',
        'Cls30', 'Cls31', 'Cls32', 'Cls33', 'Cls34', 'Cls35', 'Cls36', 'Cls37',
        'Cls38', 'Cls39', 'Cls40', 'Cls41', 'Cls42', 'Cls43', 'Cls44', 'Cls45', 'Cls46', 'Cls47']

def xywh2xyxy(x):
    '''
    Box (center x, center y, width, height) to (x1, y1, x2, y2)
    '''
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xyxy2xywh(box):
    '''
    Box (left_top x, left_top y, right_bottom x, right_bottom y) to (left_top x, left_top y, width, height)
    '''
    box[:, 2:] = box[:, 2:] - box[:, :2]
    return box

def box_area(boxes :array):
    """
    :param boxes: [N, 4]
    :return: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(box1 :array, box2: array):
    """
    :param box1: [N, 4]
    :param box2: [M, 4]
    :return: [N, M]
    """
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM

def numpy_nms(boxes :array, scores :array, iou_threshold :float):

    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)

        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]

    keep = np.array(keep)
    return keep

def NMS(dets, thresh):
    '''
    单类NMS算法
    dets.shape = (N, 5), (left_top x, left_top y, right_bottom x, right_bottom y, Scores)
    '''
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    areas = (y2-y1+1) * (x2-x1+1)
    scores = dets[:,4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size >0:
        i = index[0]       # every time the first is the biggst, and add it directly
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22-x11+1)    # the weights of overlap
        h = np.maximum(0, y22-y11+1)    # the height of overlap
        overlaps = w*h
        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
        idx = np.where(ious<=thresh)[0]
        index = index[idx+1]   # because index start from 1
 
    return dets[keep]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess_img(img, target_shape:tuple=None, div_num=255, means:list=[0.485, 0.456, 0.406], stds:list=[0.229, 0.224, 0.225]):
    '''
    图像预处理:
    target_shape: 目标shape
    div_num: 归一化除数
    means: len(means)==图像通道数，通道均值, None不进行zscore
    stds: len(stds)==图像通道数，通道方差, None不进行zscore
    '''
    img_processed = np.copy(img)
    # resize
    if target_shape:
        # img_processed = cv2.resize(img_processed, target_shape)
        img_processed = letterbox(img_processed, target_shape, stride=None, auto=False)[0]

    img_processed = img_processed.astype(np.float32)
    img_processed = img_processed/div_num

    # z-score
    if means is not None and stds is not None:
        means = np.array(means).reshape(1, 1, -1)
        stds = np.array(stds).reshape(1, 1, -1)
        img_processed = (img_processed-means)/stds

    # unsqueeze
    img_processed = img_processed[None, :]

    return img_processed.astype(np.float32)
    
def convert_shape(shapes:tuple or list, int8=False):
    '''
    转化为aidlite需要的格式
    '''
    if isinstance(shapes, tuple):
        shapes = [shapes]
    out = []
    for shape in shapes:
        nums = 1 if int8 else 4
        for n in shape:
            nums *= n
        out.append(nums)
    return out

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, img_shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img_shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img_shape[0])  # y1, y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2

def detect_postprocess(prediction, img0shape, img1shape, conf_thres=0.25, iou_thres=0.45):
    '''
    检测输出后处理
    prediction: aidlite模型预测输出
    img0shape: 原始图片shape
    img1shape: 输入图片shape
    conf_thres: 置信度阈值
    iou_thres: IOU阈值
    return: list[np.ndarray(N, 5)], 对应类别的坐标框信息, xywh、conf
    '''
    h, w, _ = img1shape
    nc = prediction.shape[1] - 4
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].max(1) > conf_thres

    outputs = []
    for xi, x in enumerate(prediction):
        x = x.transpose(1, 0)
        x = x[xc[xi]]  # confidence

        result = np.split(x, (4, nc + 4, nm + nc + 4), axis=1)
        box = result[0]
        cls = result[1]
        mask = result[2]
        
        box = xywh2xyxy(box)


        conf = cls.max(1)
        conf = conf[:, np.newaxis]
        j = cls.argmax(1)[:, np.newaxis]
        x = np.concatenate((box, conf, j, mask), axis=1)

        n = x.shape[0]
        if not n:  # no boxes
            continue

        index = np.argsort(-x[:, 4])
        x = x[index]

        c = x[:, 5:6] * 0
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = numpy_nms(boxes, scores, iou_thres)

        outputs_temp = []
        for j in i:# NMS
            outputs_temp.append(x[j])

        outputs = np.array(outputs_temp)

    return outputs


    

def draw_detect_res(img, all_boxes):
    '''
    检测结果绘制
    '''
    img = img.astype(np.uint8)
    color_step = int(255/len(coco_class))
    for i in range(len(all_boxes)):
        x1, y1, x2, y2, conf, cls_id  = [t for t in all_boxes[i]]
        cv2.putText(img, f'{coco_class[int(cls_id)]}', (int(x1), int(y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(img, f'{conf:.4f}', (int(x1), int(y1-26)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f'{conf:.4f}')
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, cls_id*color_step, 255-cls_id*color_step),thickness = 2)
    return img