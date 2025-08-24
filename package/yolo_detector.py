import time
import cv2
import numpy as np
from openvino.runtime import Core
from collections import deque


class OpenvinoInference(object):
    def __init__(self, model_path):
        self.model_path = model_path
        ie = Core()
        self.model_onnx = ie.read_model(model=self.model_path)
        self.compiled_model_onnx = ie.compile_model(model=self.model_onnx, device_name="CPU")
        self.output_layer_onnx = self.compiled_model_onnx.output(0)

    def predict(self, datas):
        predict_data = self.compiled_model_onnx([datas])[self.output_layer_onnx]
        return predict_data


class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, model_path, imgsz=(640, 640)):
        # 构建openvino推理引擎
        self.openvino = OpenvinoInference(model_path)
        self.ndtype = np.single

        # 定义类别名称映射
        self.class_names = {
            0: "water",
            1: "mountain",
            2: "fire",
            3: "man",
            4: "boom",
            # 添加更多类别...
        }

        self.classes = list(self.class_names.values())  # 类别列表
        self.model_height, self.model_width = imgsz[0], imgsz[1]  # 图像resize大小
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))  # 为每个类别生成调色板

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45):
        # 前处理Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # 推理 inference
        preds = self.openvino.predict(im)

        # 后处理Post-process
        boxes = self.postprocess(preds,
                                 im0=im0,
                                 ratio=ratio,
                                 pad_w=pad_w,
                                 pad_h=pad_h,
                                 conf_threshold=conf_threshold,
                                 iou_threshold=iou_threshold,
                                 )

        return boxes

    # 确保图像是3通道BGR格式
    def ensure_3ch_bgr(self, img):
        if img is None:
            return np.zeros((self.model_height, self.model_width, 3), dtype=np.uint8)

        if len(img.shape) == 2:  # 灰度图
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:  # 单通道
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:  # 已经是3通道
            return img

    # 前处理，包括：resize, pad, HWC to CHW，BGR to RGB，归一化，增加维度CHW -> BCHW
    def preprocess(self, img):
        # 确保图像是3通道BGR格式
        img = self.ensure_3ch_bgr(img)

        # Resize and pad input image using letterbox()
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 填充

        # 再次确保图像是3通道
        img = self.ensure_3ch_bgr(img)

        # 转换为浮点数并进行归一化
        img_float = img.astype(np.float32) / 255.0

        # BGR to RGB
        img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)

        # HWC to CHW
        img_chw = np.transpose(img_rgb, (2, 0, 1))

        # 增加批次维度
        img_process = np.expand_dims(img_chw, axis=0)

        return img_process, ratio, (pad_w, pad_h)

    # 后处理，包括：阈值过滤与NMS
    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold):
        if len(preds.shape) == 3:
            # Transpose the first output: (Batch_size, xywh_conf_cls, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls)
            preds = np.transpose(preds, (0, 2, 1))

        # 将preds作为x处理
        x = preds

        # Predictions filtering by conf-threshold
        if len(x) > 0:
            # 确保处理三维数组
            if len(x.shape) == 3:
                x = x[0]  # 取出第一个batch的元素

            # 按置信度阈值过滤
            max_conf = np.max(x[..., 4:], axis=-1)
            mask = max_conf > conf_threshold
            x = x[mask]

        # 如果还有数据，创建包含box+conf+cls的矩阵
        if len(x) > 0:
            cls_conf = np.max(x[..., 4:], axis=-1)
            cls_id = np.argmax(x[..., 4:], axis=-1)
            boxes = np.concatenate([
                x[..., :4],  # bbox
                cls_conf[..., None],  # confidence
                cls_id[..., None]  # class id
            ], axis=-1)
        else:
            boxes = np.zeros((0, 6), dtype=np.float32)

        # NMS filtering
        if len(boxes) > 0:
            # 准备NMS输入
            bboxes = boxes[:, :4].tolist()
            scores = boxes[:, 4].tolist()

            # 应用NMS
            indices = cv2.dnn.NMSBoxes(bboxes, scores, conf_threshold, iou_threshold)

            if len(indices) > 0:
                # 处理单个索引的情况
                if isinstance(indices, np.ndarray) and indices.ndim == 1:
                    boxes = boxes[indices]
                # 处理数组的情况
                elif isinstance(indices, np.ndarray):
                    boxes = boxes[indices.flatten()]
                else:  # 原始格式情况
                    boxes = boxes[indices]
            else:
                boxes = np.zeros((0, 6), dtype=np.float32)

        # 重新缩放边界框
        if len(boxes) > 0:
            # 确保是一维数组
            if boxes.ndim == 1:
                boxes = boxes[np.newaxis, :]

            # Bounding boxes format change: cxcywh -> xyxy
            boxes[:, 0] -= boxes[:, 2] / 2  # x_center to x_min
            boxes[:, 1] -= boxes[:, 3] / 2  # y_center to y_min
            boxes[:, 2] += boxes[:, 0]  # width to x_max
            boxes[:, 3] += boxes[:, 1]  # height to y_max

            # Rescale bounding boxes from model shape to original image size
            boxes[:, [0, 2]] -= pad_w
            boxes[:, [1, 3]] -= pad_h
            scale = min(ratio)
            if scale > 0:
                boxes[:, :4] /= scale

            # Bounding boxes boundary clamp
            boxes[:, 0] = np.clip(boxes[:, 0], 0, im0.shape[1])  # x_min
            boxes[:, 1] = np.clip(boxes[:, 1], 0, im0.shape[0])  # y_min
            boxes[:, 2] = np.clip(boxes[:, 2], 0, im0.shape[1])  # x_max
            boxes[:, 3] = np.clip(boxes[:, 3], 0, im0.shape[0])  # y_max

            return boxes  # (x_min, y_min, x_max, y_max, conf, cls)
        else:
            return np.zeros((0, 6), dtype=np.float32)


# 全局变量
model = None
cap = None


def initialize_model(model_path='weights/best.onnx', imgsz=(640, 640)):
    """
    初始化模型和摄像头

    Args:
        model_path: 模型路径
        imgsz: 模型输入尺寸

    Returns:
        bool: 初始化是否成功
    """
    global model, cap

    try:
        # 加载模型
        print("正在加载模型...")
        model = YOLOv8(model_path, imgsz)
        print("模型加载完成！")

        # 初始化摄像头
        for camera_id in range(2):  # 尝试不同摄像头ID
            for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    print(f"使用摄像头 {camera_id}, 后端 {backend} 成功")

                    # 设置摄像头分辨率
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                    # 打印摄像头信息
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f"摄像头分辨率: {width}x{height}")
                    return True

        if not cap or not cap.isOpened():
            print("无法打开任何摄像头")
            return False

        return True
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        return False


def capture_and_detect(conf_threshold=0.25, iou_threshold=0.45):
    """
    拍摄一张图片并识别最大面积的物体

    Args:
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值

    Returns:
        str: 识别到的最大面积物体的标签，如果没有识别到则返回None
    """
    global model, cap

    if model is None or cap is None:
        print("模型或摄像头未初始化，请先调用initialize_model()")
        return None

    # 连续拍摄几帧以稳定摄像头
    for _ in range(5):
        ret, _ = cap.read()
        if not ret:
            time.sleep(0.1)

    # 读取帧
    ret, frame = cap.read()
    if not ret or frame is None:
        print("无法读取摄像头数据")
        return None

    # 确保帧是3通道
    frame = model.ensure_3ch_bgr(frame)

    # 目标检测推理
    try:
        boxes = model(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    except Exception as e:
        print(f"推理过程中出错: {str(e)}")
        return None

    # 如果没有检测到物体
    if len(boxes) == 0:
        return None

    # 计算每个框的面积并找到最大的
    max_area = 0
    max_label = None

    for box in boxes:
        if len(box) < 6:
            continue

        x_min, y_min, x_max, y_max, conf, cls_idx = box[:6]
        cls_idx = int(cls_idx)

        # 计算面积
        area = (x_max - x_min) * (y_max - y_min)

        # 更新最大面积
        if area > max_area:
            max_area = area
            if cls_idx < len(model.classes):
                max_label = model.classes[cls_idx]
            else:
                max_label = f"class_{cls_idx}"

    return max_label


def cleanup():
    """释放资源"""
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
