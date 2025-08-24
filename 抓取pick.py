import cv2  # OpenCV计算机视觉库，用于图像处理和摄像头操作
import time  # 时间模块，用于添加延时
import numpy as np  # NumPy数值计算库，用于数组操作
from xgolib import XGO  # XGO机器狗控制库


class DogVisionSystem:
    """
    机器狗视觉系统类
    提供颜色识别、目标追踪、抓取和放置功能
    """

    def __init__(self):
        """
        初始化机器狗视觉系统
        设置机器狗初始姿态、颜色识别参数和摄像头
        """
        # 初始化机器狗连接（通过串口/dev/ttyAMA0）
        self.dog = XGO("/dev/ttyAMA0")

        # 设置机器狗初始姿态
        self.dog.attitude('p', 40)  # 设置俯仰角为40度，让摄像头向下看
        self.dog.arm(-90, 90)  # 设置机械臂初始位置

        # 定义HSV颜色空间中各种颜色的阈值范围
        self.color_dist = {
            # 红色需要两个范围，因为红色在HSV色轮的两端
            'red': {
                'Lower1': np.array([0, 60, 60]),  # 红色范围1下界 (H, S, V)
                'Upper1': np.array([6, 255, 255]),  # 红色范围1上界
                'Lower2': np.array([170, 60, 60]),  # 红色范围2下界
                'Upper2': np.array([180, 255, 255])  # 红色范围2上界
            },
            # 蓝色范围
            'blue': {
                'Lower': np.array([100, 80, 46]),  # 蓝色下界
                'Upper': np.array([124, 255, 255])  # 蓝色上界
            },
            # 绿色范围
            'green': {
                'Lower': np.array([35, 100, 100]),  # 绿色下界
                'Upper': np.array([85, 255, 255])  # 绿色上界
            },
        }

        # 初始化控制参数
        self.threshold_range = 20  # 水平方向允许的偏差像素数
        self.adjustment_count = 0  # 调整计数器，记录连续调整次数
        self.vertical_threshold = 150  # 垂直方向的距离阈值
        self.max_attempts = 7  # 最大尝试次数，达到后认为已对准

        # 初始化摄像头（设备号0）
        self.cap = cv2.VideoCapture(0)

        # 设置机器狗移动速度为低速
        self.dog.pace('low')

        # 创建OpenCV显示窗口
        cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)

        print("机器狗视觉系统初始化完成")

    def __del__(self):
        """
        析构函数，对象销毁时自动调用
        确保资源得到正确释放
        """
        self.cleanup()

    def cleanup(self):
        """
        清理函数，释放所有占用的资源
        包括摄像头、OpenCV窗口和机器狗复位
        """
        # 释放摄像头资源
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()

        # 重置机器狗到初始状态
        self.dog.reset()
        print("资源已释放")

    def apply_morphology(self, mask, kernel_size=(7, 7), iterations=2):
        """
        对二值化掩码图像应用形态学操作
        用于去除噪声和填补空洞，改善目标检测效果

        参数:
            mask: 输入的二值化掩码图像
            kernel_size: 形态学操作的核大小，默认7x7
            iterations: 操作迭代次数，默认2次

        返回:
            处理后的掩码图像
        """
        # 创建矩形结构元素（核）
        kernel = np.ones(kernel_size, np.uint8)

        # 腐蚀操作：去除小的噪声点，但会让目标变小
        mask_erode = cv2.erode(mask, kernel, iterations=iterations)

        # 膨胀操作：恢复目标大小并填补空洞
        mask_dilate = cv2.dilate(mask_erode, kernel, iterations=iterations)

        return mask_dilate

    def zhuaqu(self):
        """
        执行抓取物体的完整动作序列
        包括调整姿态、移动机械臂、夹取物体等步骤

        返回:
            操作成功返回True
        """
        print("开始执行抓取操作...")

        # 第一步：调整机身姿态，俯仰40度便于观察和操作
        self.dog.attitude('p', 40)

        # 第二步：张开机械爪，准备夹取
        self.dog.claw(0)  # 0表示完全张开

        # 第三步：机械臂运动序列
        # 将机械臂末端移到基座正上方100mm处
        self.dog.arm(0, 100)
        time.sleep(0.5)  # 等待动作完成

        # 机械臂大臂垂直，小臂水平
        self.dog.arm(90, 90)
        time.sleep(0.5)

        # 将机械臂末端移到摄像头正前方
        self.dog.arm(100, 0)
        time.sleep(0.5)

        # 机械臂下探到抓取位置
        self.dog.arm(100, -60)  # 负值表示向下
        time.sleep(2)  # 等待2秒确保到位

        # 第四步：夹住物体
        self.dog.claw(230)  # 230表示夹紧程度
        time.sleep(1)  # 等待夹紧完成

        # 第五步：收回机械臂
        # 先回到正前方
        self.dog.arm(100, 0)
        time.sleep(0.5)

        # 再收回到安全位置
        self.dog.arm(0, 90)
        time.sleep(0.5)

        # 第六步：恢复正常站立姿态
        self.dog.attitude('p', 0)  # 俯仰角归零
        time.sleep(1)

        print("抓取操作完成")
        return True

    def fangzhi(self):
        """
        执行放置物体的完整动作序列
        与抓取过程类似，但最后是放开而不是夹紧

        返回:
            操作成功返回True
        """
        print("开始执行放置操作...")

        # 第一步：调整机身姿态
        self.dog.attitude('p', 40)

        # 第二步：机械臂运动到放置位置
        # 初始位置调整（参数140可能是为了不同的放置距离）
        self.dog.arm(140, 0)
        time.sleep(0.5)

        # 标准运动序列
        self.dog.arm(90, 90)
        time.sleep(0.5)

        self.dog.arm(100, 0)
        time.sleep(0.5)

        # 下探到放置位置
        self.dog.arm(100, -60)
        time.sleep(2)

        # 第三步：松开机械爪，放下物体
        self.dog.claw(10)  # 10表示微微张开，确保物体掉落
        time.sleep(1)

        # 第四步：收回机械臂
        self.dog.arm(100, 0)
        time.sleep(0.5)

        # 第五步：完全复位系统
        self.dog.attitude('p', 0)  # 恢复水平姿态
        self.dog.reset()  # 重置所有关节
        self.dog.claw(100)  # 设置爪子到中等开合度
        time.sleep(1)

        print("放置操作完成")
        return True

    def track_target(self, target_color='blue'):
        """
        追踪指定颜色的目标物体，但不执行抓取操作
        持续调整机器狗位置直到目标在合适的抓取位置

        参数:
            target_color: 要追踪的目标颜色 ('red', 'blue', 'green')

        返回:
            成功追踪到目标返回True，用户中断或失败返回False
        """
        # 重置调整计数器
        self.adjustment_count = 0

        # 主追踪循环
        while True:
            # 从摄像头读取一帧图像
            ret, frame = self.cap.read()
            if not ret:
                print("摄像头读取失败")
                return False

            # 图像预处理步骤
            # 1. 高斯模糊去噪声
            gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # 2. 转换到HSV颜色空间（更适合颜色检测）
            hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)

            # 3. 双边滤波和中值滤波进一步降噪
            # 中值滤波去除椒盐噪声
            median_filtered = cv2.medianBlur(hsv, 5)
            # 双边滤波保持边缘同时平滑
            hsv_filtered = cv2.bilateralFilter(median_filtered, 9, 75, 75)

            # 根据目标颜色创建相应的颜色掩码
            if target_color == 'blue':
                # 蓝色只需要一个范围
                mask = cv2.inRange(hsv_filtered,
                                   self.color_dist['blue']['Lower'],
                                   self.color_dist['blue']['Upper'])
            elif target_color == 'red':
                # 红色需要两个范围并合并（因为红色跨越HSV色轮的0度）
                mask_red1 = cv2.inRange(hsv_filtered,
                                        self.color_dist['red']['Lower1'],
                                        self.color_dist['red']['Upper1'])
                mask_red2 = cv2.inRange(hsv_filtered,
                                        self.color_dist['red']['Lower2'],
                                        self.color_dist['red']['Upper2'])
                mask = cv2.bitwise_or(mask_red1, mask_red2)
            else:  # 默认为绿色
                mask = cv2.inRange(hsv_filtered,
                                   self.color_dist['green']['Lower'],
                                   self.color_dist['green']['Upper'])

            # 应用形态学操作优化掩码
            processed_mask = self.apply_morphology(mask)

            # 查找轮廓
            # findContours返回轮廓列表，[-2]获取轮廓数据
            cnts = cv2.findContours(processed_mask,
                                    cv2.RETR_EXTERNAL,  # 只检测外轮廓
                                    cv2.CHAIN_APPROX_SIMPLE  # 压缩轮廓点
                                    )[-2]

            # 根据检测到的轮廓调整机器狗位置
            if self.adjust_position(frame, cnts):
                # 成功定位目标，点亮蓝色LED指示
                self.dog.rider_led(3, [0, 0, 255])  # RGB: 红, 绿, 蓝
                return True

            # 显示处理后的图像（带有标记）
            cv2.imshow('camera', frame)

            # 检测用户输入，按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户中断操作")
                return False

    def adjust_position(self, frame, cnts):
        """
        根据检测到的目标轮廓调整机器狗位置
        实现目标居中和距离控制

        参数:
            frame: 当前帧图像，用于绘制标记
            cnts: 检测到的轮廓列表

        返回:
            如果目标已正确定位返回True，否则返回False
        """
        # 计算图像中心点坐标
        img_center_x = frame.shape[1] // 2  # 图像宽度的一半
        img_center_y = frame.shape[0] // 2  # 图像高度的一半

        # 在图像中心绘制白色圆点作为参考
        cv2.circle(frame, (img_center_x, img_center_y), 5, (255, 255, 255), -1)

        # 遍历所有检测到的轮廓
        for c in cnts:
            # 过滤掉面积太小的轮廓（可能是噪声）
            if cv2.contourArea(c) < 500:
                continue

            # 获取轮廓的外接矩形
            x, y, w, h = cv2.boundingRect(c)

            # 计算矩形中心点
            rect_center_x = x + w // 2
            rect_center_y = y + h // 2

            # 计算目标中心与图像中心的偏差
            horizontal_diff = rect_center_x - img_center_x  # 水平偏差
            vertical_diff = rect_center_y - img_center_y  # 垂直偏差

            # 在图像上绘制检测结果的可视化标记
            # 绿色矩形框标记检测到的目标
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 红色圆点标记目标中心
            cv2.circle(frame, (rect_center_x, rect_center_y), 5, (0, 0, 255), -1)

            # 蓝色线段连接图像中心和目标中心，显示偏差
            cv2.line(frame, (img_center_x, img_center_y),
                     (rect_center_x, rect_center_y), (255, 0, 0), 2)

            # 位置控制逻辑
            # 检查是否已经正确定位：水平偏差小且垂直距离足够（目标在下方）
            if (abs(horizontal_diff) < self.threshold_range and
                    vertical_diff > self.vertical_threshold):

                # 停止移动
                self.dog.stop()

                # 增加调整计数器
                self.adjustment_count += 1

                # 如果连续多次都在正确位置，认为定位成功
                if self.adjustment_count >= self.max_attempts:
                    return True

            else:
                # 位置不正确，重置计数器并开始调整
                self.adjustment_count = 0

                # 水平位置调整
                if abs(horizontal_diff) >= self.threshold_range:
                    # 根据偏差方向决定移动方向
                    # 如果目标在右边(horizontal_diff > 0)，机器狗向右移动(y=-1)
                    # 如果目标在左边(horizontal_diff < 0)，机器狗向左移动(y=1)
                    move_value = -1 if horizontal_diff > 0 else 1
                    self.dog.move("y", move_value)

                # 垂直距离调整
                if vertical_diff < self.vertical_threshold:
                    # 目标太近或在上方，后退
                    self.dog.move("x", 2)
                else:
                    # 目标距离合适但可能需要微调，轻微前进
                    self.dog.move("x", -2)

        # 没有达到定位条件，返回False继续调整
        return False
