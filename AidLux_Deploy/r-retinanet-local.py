# aidlux相关
from cvs import *
import aidlite_gpu
import cv2
import numpy as np

def get_std_rect(points):
    npoints = points.shape[0]
    if len(points.shape) != 2 or points.shape[1] !=2 or npoints == 0:
        return None, None, None, None
    x_coord = points[:, 0]
    y_coord = points[:, 1]
    print("y_coord.shape: {}".format(y_coord.shape))
    x = x_coord.min()
    y = y_coord.min()
    w = x_coord.max() - x
    h = y_coord.max() - y
    return int(x), int(y),int(w), int(h)

def nms(dets, thresh):
    ws = dets[:, 2] - dets[:, 0]
    hs = dets[:, 3] - dets[:, 1]
    xx = dets[:, 0] + ws * 0.5
    yy = dets[:, 1] + hs * 0.5
    tt = dets[:, 4]
    areas = ws * hs

    scores = dets[:, 5]
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int_)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            rtn, contours = cv2.rotatedRectangleIntersection(
                ((xx[i], yy[i]), (ws[i], hs[i]), tt[i]),
                ((xx[j], yy[j]), (ws[j], hs[j]), tt[j])
            )
            if rtn == 1:
                inter = np.round(np.abs(cv2.contourArea(contours)))
            elif rtn == 2:
                inter = min(areas[i], areas[j])
            else:
                inter = 0.0
            ovr = inter / (areas[i] + areas[j] - inter + 1e-6)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep

def clip_boxes(boxes, ims):
    _, _, h, w = ims.shape
    boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
    boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)
    boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, w)
    boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, h)
    return boxes

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel(),
        np.zeros(shift_x.ravel().shape)
    )).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 5)) + shifts.reshape((1, K, 5)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 5))
    return all_anchors

def generate_anchors(base_size, ratios, scales, rotations):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    num_anchors = len(ratios) * len(scales) * len(rotations)  # 所生成总的anchor数
    # initialize output anchors
    anchors = np.zeros((num_anchors, 5))
    # scale base_size
    anchors[:, 2:4] = base_size * np.tile(scales, (2, len(ratios) * len(rotations))).T # 假设base_size=16, 此处生成shape(5, 2)值全为16的array
    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]  # shape(5, )
    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales) * len(rotations)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales) * len(rotations))
    # add rotations
    anchors[:, 4] = np.tile(np.repeat(rotations, len(scales)), (1, len(ratios))).T[:, 0]
    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0:3:2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1:4:2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors

class Anchors():
    def __init__(self,
                 pyramid_levels=None,
                 strides=None,
                 sizes=None,
                 ratios=None,  # np.array([0.2, 0.5, 1, 2, 5])
                 scales=None,
                 rotations=None):
        super(Anchors, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.strides =  strides
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales
        self.rotations = rotations
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 1) for x in self.pyramid_levels]  # [16, 32, 64, 126, 256]
        if ratios is None:
            self.ratios = np.array([1])
        if scales is None:
            self.scales = np.array([2 ** 0])
        if rotations is None:
            self.rotations = np.array([0])
        self.num_anchors = len(self.scales) * len(self.ratios) * len(self.rotations)

    def __call__(self, ims):
        ims_shape = np.array(ims.shape[2:])  # feature map的h和w
        # ims_shape = np.array([640, 800])  # [SAI-KEY] 生成ONNX时，需固定
        image_shapes = [(ims_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]  # FPN输出的各feature map的分辨率
                                                                                            # 输入640, 输出80*80, 40*40, 20*20, 10*10, 5*5
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 5)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):  # p=3、4、5、6、7
            anchors = generate_anchors(
                base_size=self.sizes[idx],  # [ 16,  32, 64, 128, 256]
                ratios=self.ratios,         # [0.2, 0.5,  1,   2,   5]
                scales=self.scales,         # 1
                rotations=self.rotations    # 0
            )
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        all_anchors = np.tile(all_anchors, (ims.shape[0], 1, 1))
        all_anchors = all_anchors.astype(np.float32)
        return all_anchors


class BoxCoder():
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """
    def __init__(self, weights=(10., 10., 10., 5., 15.)):
        self.weights = weights

    def decode(self, boxes, deltas, mode='xywht'):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        widths = np.clip(widths, 1, 2000)
        heights = np.clip(heights, 1, 2000)
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights
        thetas = boxes[:, :, 4]  # angles

        wx, wy, ww, wh, wt = self.weights
        dx = deltas[:, :, 0] / wx
        dy = deltas[:, :, 1] / wy
        dw = deltas[:, :, 2] / ww
        dh = deltas[:, :, 3] / wh
        dt = deltas[:, :, 4] / wt

        pred_ctr_x = ctr_x if 'x' not in mode else ctr_x + dx * widths
        pred_ctr_y = ctr_y if 'y' not in mode else ctr_y + dy * heights
        pred_w = widths if 'w' not in mode else np.exp(dw) * widths
        pred_h = heights if 'h' not in mode else np.exp(dh) * heights
        pred_t = thetas if 't' not in mode else np.arctan(np.tan(thetas / 180.0 * np.pi) + dt) / np.pi * 180.0

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = np.stack([
            pred_boxes_x1,
            pred_boxes_y1,
            pred_boxes_x2,
            pred_boxes_y2,
            pred_t], axis=2
        )
        return pred_boxes

def sort_corners(quads):
    sorted = np.zeros(quads.shape, dtype=np.float32)
    for i, corners in enumerate(quads):  # corners: xyxyxyxy
        corners = corners.reshape(4, 2)
        centers = np.mean(corners, axis=0)  # 中心点
        corners = corners - centers
        cosine = corners[:, 0] / np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
        cosine = np.minimum(np.maximum(cosine, -1.0), 1.0)
        thetas = np.arccos(cosine) / np.pi * 180.0
        indice = np.where(corners[:, 1] > 0)[0]
        thetas[indice] = 360.0 - thetas[indice]
        corners = corners + centers
        corners = corners[thetas.argsort()[::-1], :]
        corners = corners.reshape(8)
        dx1, dy1 = (corners[4] - corners[0]), (corners[5] - corners[1])
        dx2, dy2 = (corners[6] - corners[2]), (corners[7] - corners[3])
        slope_1 = dy1 / dx1 if dx1 != 0 else np.iinfo(np.int32).max
        slope_2 = dy2 / dx2 if dx2 != 0 else np.iinfo(np.int32).max
        if slope_1 > slope_2:
            if corners[0] < corners[4]:
                first_idx = 0
            elif corners[0] == corners[4]:
                first_idx = 0 if corners[1] < corners[5] else 2
            else:
                first_idx = 2
        else:
            if corners[2] < corners[6]:
                first_idx = 1
            elif corners[2] == corners[6]:
                first_idx = 1 if corners[3] < corners[7] else 3
            else:
                first_idx = 3
        for j in range(4):
            idx = (first_idx + j) % 4
            sorted[i, j*2] = corners[idx*2]
            sorted[i, j*2+1] = corners[idx*2+1]
    return sorted

def rbox_2_quad(rboxes, mode='xyxya'):
    if len(rboxes.shape) == 1:
        rboxes = rboxes[np.newaxis, :]
    if rboxes.shape[0] == 0:
        return rboxes
    quads = np.zeros((rboxes.shape[0], 8), dtype=np.float32)
    for i, rbox in enumerate(rboxes):
        if len(rbox!=0):
            if mode == 'xyxya':
                w = rbox[2] - rbox[0]
                h = rbox[3] - rbox[1]
                x = rbox[0] + 0.5 * w
                y = rbox[1] + 0.5 * h
                theta = rbox[4]
            elif mode == 'xywha':
                x = rbox[0]
                y = rbox[1]
                w = rbox[2]
                h = rbox[3]
                theta = rbox[4]
            quads[i, :] = cv2.boxPoints(((x, y), (w, h), theta)).reshape((1, 8))

    return quads

def decoder(ims, anchors, cls_score, bbox_pred, thresh=0.6, nms_thresh=0.2, test_conf=None):
        if test_conf is not None:
            thresh = test_conf
        bboxes = BoxCoder().decode(anchors, bbox_pred, mode='xywht')
        bboxes = clip_boxes(bboxes, ims)
        scores = cls_score.max(2, keepdims=True)
        keep = (scores >= thresh)[0, :, 0]
        if keep.sum() == 0:
            return [np.zeros(1), np.zeros(1), np.zeros(1, 5)]
        scores = scores[:, keep, :]
        anchors = anchors[:, keep, :]
        cls_score = cls_score[:, keep, :]
        bboxes = bboxes[:, keep, :]
        # NMS
        anchors_nms_idx = nms(np.concatenate([bboxes, scores], axis=2)[0, :, :], nms_thresh)
        nms_scores = cls_score[0, anchors_nms_idx, :].max(axis=1)
        nms_class = cls_score[0, anchors_nms_idx, :].argmax(axis=1)
        output_boxes = np.concatenate([
            bboxes[0, anchors_nms_idx, :],
            anchors[0, anchors_nms_idx, :]],
            axis=1
        )
        return [nms_scores, nms_class, output_boxes]



def process_img(img, target_size=640, max_size=2000, multiple=32, keep_ratio=True, NCHW=True, ToTensor=True):
    '''
    图像与处理
    '''
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # resize with keep_ratio
    if keep_ratio:
        im_scale = float(target_size) / float(im_size_min)  
        if np.round(im_scale * im_size_max) > max_size:     
            im_scale = float(max_size) / float(im_size_max)
        im_scale_x = np.floor(img.shape[1] * im_scale / multiple) * multiple / img.shape[1]
        im_scale_y = np.floor(img.shape[0] * im_scale / multiple) * multiple / img.shape[0]
        image_resized = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)
        im_scales = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
        im = image_resized / 255.0  # np.float64
        im = im.astype(np.float32)
        PIXEL_MEANS =(0.485, 0.456, 0.406)    # RGB  format mean and variances
        PIXEL_STDS = (0.229, 0.224, 0.225)
        im -= np.array(PIXEL_MEANS)
        im /= np.array(PIXEL_STDS)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # BGR2RGB
        if NCHW:
            im = np.transpose(im, (2, 0, 1)).astype(np.float32)  # [SAI-KEY] TensorFlow use input with NHWC.
        im = im[np.newaxis, ...]
        if ToTensor:
            im = torch.from_numpy(im)
        return im, im_scales
    else:
        return None

if __name__=="__main__":
    '''
    读取本地图片
    '''
    image_path = "/home/code/AidLux_Deploy/samples/000001.jpg"
    cap = cvs.VideoCapture(image_path)
    img = cap.read()
    print(img.shape)
    im, im_scales = process_img(img, NCHW=False, ToTensor=False)  # im: NHWC

    ''' 定义输入输出shape '''
    in_shape = [1 * 640 * 800 * 3 * 4]  # HWC, float32
    out_shape = [1 * 53325 * 8 * 4]  # 8400: total cells, 52 = 48(num_classes) + 4(xywh), float32
    # out_shape = [1 * 55425 * 8 * 4]  # 8400: total cells, 52 = 48(num_classes) + 4(xywh), float32

    ''' AidLite初始化 '''
    aidlite = aidlite_gpu.aidlite()
    ''' 加载R-RetinaNet模型 '''
    tflite_model = '/home/code/AidLux_Deploy/models/r-retinanet-statedict.tflite'
    res = aidlite.ANNModel(tflite_model, in_shape, out_shape, 4, -1) # Infer on -1: cpu, 0: gpu, 1: mixed, 2: dsp

    ''' 设定输入输出 '''
    aidlite.setInput_Float32(im, 800, 640)

    ''' 启动推理 '''
    aidlite.invoke()

    ''' 捕获输出 '''
    preds = aidlite.getOutput_Float32(0)
    # preds = preds.reshape(1, 8, 53325)
    preds = preds.reshape(1, 8, (int)(preds.shape[0]/8))
    output = np.transpose(preds, (0, 2, 1))

    ''' 创建Anchor '''
    im_anchor = np.transpose(im, (0, 3, 1, 2)).astype(np.float32)
    anchors_list = []
    anchor_generator = Anchors(ratios = np.array([0.2, 0.5, 1, 2, 5]))
    original_anchors = anchor_generator(im_anchor)   # (bs, num_all_achors, 5)
    anchors_list.append(original_anchors)

    ''' 解算输出 '''
    decode_output = decoder(im_anchor, anchors_list[-1], output[..., 5:8], output[..., 0:5], thresh=0.5, nms_thresh=0.2, test_conf=None)
    for i in range(len(decode_output)):
        print("dim({}), shape: {}".format(i, decode_output[i].shape))

    ''' 重构输出 '''
    scores = decode_output[0].reshape(-1, 1)
    classes = decode_output[1].reshape(-1, 1)
    boxes = decode_output[2]
    boxes[:, :4] = boxes[:, :4] / im_scales
    if boxes.shape[1] > 5:   
        boxes[:, 5:9] = boxes[:, 5:9] / im_scales
    dets = np.concatenate([classes, scores, boxes], axis=1)

    ''' 过滤类别 '''
    keep = np.where(classes > 0)[0]
    dets =  dets[keep, :]

    ''' 转换坐标('xyxya'->'xyxyxyxy') '''
    res = sort_corners(rbox_2_quad(dets[:, 2:]))

    ''' 评估绘图 '''
    for k in range(dets.shape[0]):
        cv2.line(img, (int(res[k, 0]), int(res[k, 1])), (int(res[k, 2]), int(res[k, 3])), (0, 255, 0), 3)
        cv2.line(img, (int(res[k, 2]), int(res[k, 3])), (int(res[k, 4]), int(res[k, 5])), (0, 255, 0), 3)
        cv2.line(img, (int(res[k, 4]), int(res[k, 5])), (int(res[k, 6]), int(res[k, 7])), (0, 255, 0), 3)
        cv2.line(img, (int(res[k, 6]), int(res[k, 7])), (int(res[k, 0]), int(res[k, 1])), (0, 255, 0), 3)
    cv2.imwrite("/home/code/AidLux_Deploy/samples/00_detected_image.jpg", img)

    ''' 将绝缘子旋转至水平 '''
    t_center = ((dets[0, 4]+dets[0, 2])/2, (dets[0,5]+dets[0,3])/2)
    t_angle = dets[0, 6]
    t_height, t_width = img.shape[:2]
    rotate_matrix = cv2.getRotationMatrix2D(center=t_center, angle=t_angle, scale=1)
    rotated_image = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(t_width, t_height))
    
    ''' 转换旋转后的坐标 '''
    new_coord = np.zeros((dets.shape[0], 4, 2), dtype=np.float)

    ''' 当存在多根绝缘子, 以其中一条为例进行后处理 '''
    k = 0
    new_coord[k, 0] = np.squeeze(np.dot(rotate_matrix, np.array([[res[k, 0]], [res[k, 1]], [1]])))
    new_coord[k, 1] = np.squeeze(np.dot(rotate_matrix, np.array([[res[k, 2]], [res[k, 3]], [1]])))
    new_coord[k, 2] = np.squeeze(np.dot(rotate_matrix, np.array([[res[k, 4]], [res[k, 5]], [1]])))
    new_coord[k, 3] = np.squeeze(np.dot(rotate_matrix, np.array([[res[k, 6]], [res[k, 7]], [1]])))

    ''' 获取标准外接矩形 '''
    (x, y, w, h) = get_std_rect(new_coord[k])

    ''' 提取ROI图像 '''
    roi_image = rotated_image[y:(y+h), x:(x+w)]
    ''' 灰度图 '''
    gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    ''' 二值化 '''
    retval, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    ''' 创建一个5*5的值为1的卷积核 '''
    kernel = np.ones((5, 5), np.uint8)
    ''' 腐蚀运算, 迭代1次 '''
    erode_image = cv2.erode(binary_image, kernel, iterations=1)

    ''' 存储本地评估 '''
    cv2.imwrite("/home/code/AidLux_Deploy/samples/01_rotated_image.jpg", rotated_image)
    cv2.imwrite("/home/code/AidLux_Deploy/samples/02_roi_image.jpg", roi_image)
    cv2.imwrite("/home/code/AidLux_Deploy/samples/03_binary_image.jpg", binary_image)
    cv2.imwrite("/home/code/AidLux_Deploy/samples/04_erode_image.jpg", erode_image)
