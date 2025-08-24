import math
import time
import cv2
import numpy as np
import xgolib
from move_lib import *


# PID控制器类
class PIDController:
    def __init__(self, kp=0.8, ki=0.01, kd=0.01, max_output=None, min_output=None):
        """
        PID控制器初始化
        Args:
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
            max_output: 最大输出限制
            min_output: 最小输出限制
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.min_output = min_output

        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error, dt=None):
        """
        PID控制器更新
        Args:
            error: 当前误差值
            dt: 时间间隔，如果为None则自动计算
        Returns:
            控制输出值
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time

        # 防止dt为0或过小
        if dt <= 0:
            dt = 0.01

        # 比例项
        proportional = self.kp * error

        # 积分项
        self.integral += error * dt
        integral = self.ki * self.integral

        # 微分项
        derivative = self.kd * (error - self.previous_error) / dt

        # PID输出
        output = proportional + integral + derivative

        # 输出限制
        if self.max_output is not None:
            output = min(output, self.max_output)
        if self.min_output is not None:
            output = max(output, self.min_output)

        # 更新历史值
        self.previous_error = error
        self.last_time = current_time

        return output

    def reset(self):
        """重置PID控制器"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


# 全局PID控制器实例
pid_x = PIDController(kp=0.5, ki=0.01, kd=0.1, max_output=15, min_output=-15)  # X轴位置控制
pid_y = PIDController(kp=0.5, ki=0.01, kd=0.1, max_output=15, min_output=-15)  # Y轴位置控制

# 颜色检测相关常量
lower_yellow = np.array([17, 231, 227])  # 黄色范围的下界
upper_yellow = np.array([23, 255, 255])  # 黄色范围的上界

# 红色颜色范围定义（HSV色彩空间中红色分布在两个区间）
red_lower1 = [0, 50, 50]  # 第一个红色范围的下限
red_upper1 = [10, 255, 255]  # 第一个红色范围的上限
red_lower2 = [170, 50, 50]  # 第二个红色范围的下限
red_upper2 = [180, 255, 255]  # 第二个红色范围的上限
color_ranges_red = [[red_lower1, red_upper1], [red_lower2, red_upper2]]


def filter_img(frame, color):
    """
    从给定的视频帧中过滤出特定颜色范围内的图像部分
    Args:
        frame: 输入的图像帧
        color: 颜色范围，格式为[下限, 上限]
    Returns:
        过滤后的图像掩码
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_lower = np.array(color[0])
    color_upper = np.array(color[1])
    mask = cv2.inRange(hsv, color_lower, color_upper)
    img_mask = cv2.bitwise_and(frame, frame, mask=mask)
    return img_mask


def filter_img_red(frame, color_ranges):
    """
    专门用于红色检测的图像过滤函数
    由于红色在HSV色彩空间中分布在两个区间，需要特殊处理
    Args:
        frame: 输入图像帧
        color_ranges: 红色的颜色范围列表
    Returns:
        过滤后的红色图像掩码
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = [cv2.inRange(hsv, np.array(color_range[0]), np.array(color_range[1])) for color_range in color_ranges]
    combined_mask = cv2.bitwise_or(*masks)
    img_mask = cv2.bitwise_and(frame, frame, mask=combined_mask)
    return img_mask


def detect_contours(frame):
    """
    在图像中检测轮廓，并返回这些轮廓以及用于检测轮廓的边缘图像
    Args:
        frame: 输入图像帧
    Returns:
        contours: 检测到的轮廓列表
        edges: 边缘检测后的图像
    """
    # 将彩色图像转换为灰度图像
    edges = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 形态学操作：先膨胀后腐蚀，用于连接断开的轮廓和填充小洞
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=3)  # 膨胀操作
    edges = cv2.erode(edges, kernel, iterations=3)  # 腐蚀操作

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, edges


def detect_color():
    """
    检测地面颜色，返回检测到的主要颜色
    Returns:
        color: 检测到的颜色字符串 ('red', 'green', 'blue')
    """
    # 调整机器狗姿态为俯视角度
    dog.attitude("p", 15)
    time.sleep(0.5)

    # 初始化摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 定义各种颜色的HSV范围
    lower_red_1 = np.array([0, 50, 50])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 50, 50])
    upper_red_2 = np.array([180, 255, 255])
    lower_green = np.array([65, 120, 0])
    upper_green = np.array([80, 255, 255])
    lower_blue = np.array([90, 50, 0])
    upper_blue = np.array([110, 255, 255])

    while 1:
        # 获取一帧图像
        ret, frame = cap.read()

        # 只识别下面70%区域，去除上方30%可能的干扰
        height, width, _ = frame.shape
        start_row = int(height * 0.3)  # 下部70%区域的起始行位置
        start_col = int(width * 0.2)  # 左侧去掉20%
        end_col = int(width * 0.8)  # 右侧去掉20%

        # 裁剪图像到感兴趣区域
        cropped_image = frame[start_row:height, start_col:end_col]

        # 将BGR图像转换为HSV色彩空间
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

        # 创建各种颜色的掩膜
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)  # 合并两个红色掩膜
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # 对掩膜进行形态学操作，去除噪声
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        # 计算每个颜色掩膜中非零像素的数量
        blue_count = cv2.countNonZero(blue_mask)
        red_count = cv2.countNonZero(red_mask)
        green_count = cv2.countNonZero(green_mask)

        # 确定哪种颜色占据更多的像素
        max_count = max(blue_count, red_count, green_count)
        if max_count == blue_count:
            color = 'blue'
            break
        elif max_count == red_count:
            color = 'red'
            break
        else:
            color = 'green'
            break

    cap.release()
    cv2.destroyAllWindows()
    dog.reset()
    time.sleep(0.5)
    return color


def detect_block(contours, frame):
    """
    从检测到的轮廓中找出最大的块状物体，并返回其相关参数
    Args:
        contours: 轮廓列表
        frame: 原始图像帧
    Returns:
        flag: 是否检测到有效块状物体
        length: 物体长度
        width: 物体宽度
        angle: 物体角度
        s_x: 物体中心X坐标（屏幕坐标系）
        s_y: 物体中心Y坐标（屏幕坐标系）
        frame: 绘制了检测结果的图像帧
    """
    flag = False
    length, width, angle, s_x, s_y = 0, 0, 0, 0, 0

    for i in range(0, len(contours)):
        # 过滤掉面积太小的轮廓
        if cv2.contourArea(contours[i]) < 5000:
            continue

        # 计算最小外接矩形
        rect = cv2.minAreaRect(contours[i])

        # 只处理第一个满足条件的轮廓
        if not flag:
            # 根据角度确定长宽
            if rect[2] > 45:
                length = rect[1][0]
                width = rect[1][1]
                angle = rect[2]
            else:
                length = rect[1][1]
                width = rect[1][0]
                angle = rect[2]

            # 获取物体中心坐标（注意：这里X和Y坐标有交换）
            s_x = rect[0][1]  # s_代表屏幕坐标系
            s_y = rect[0][0]
            flag = True

            # 绘制最小外接矩形
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 5)
            break

    return flag, length, width, angle, s_x, s_y, frame


def adjust(m_angle, m_x, m_y, opposite_yaw, des_x=1200, des_y=980):
    """
    使用PID控制算法调整机器人位置到目标点
    Args:
        m_angle: 物体角度
        m_x: 当前X坐标
        m_y: 当前Y坐标
        opposite_yaw: 目标朝向角度
        des_x: 目标X坐标
        des_y: 目标Y坐标
    Returns:
        True表示到达目标位置并完成抓取，False表示还需要继续调整
    """
    origin_yaw = opposite_yaw

    # 计算位置误差
    err_x = des_x - m_x
    err_y = des_y - m_y
    print(f"位置误差 - X: {err_x}, Y: {err_y}")
    # 使用PID控制器计算控制输出
    control_x = pid_x.update(err_x)
    control_y = pid_y.update(err_y)
    print(f"PID控制输出 - X: {control_x:.2f}, Y: {control_y:.2f}")
    # Y轴方向的控制（左右移动）
    if abs(err_x) < 70:
        if abs(err_y) < 40:
            # 到达目标位置，执行抓取
            grasp()
            return True
        else:
            # 细微Y轴调整，使用PID控制
            dog.gait_type("slow_trot")
            # 将PID输出转换为实际的移动参数
            move_speed = max(8, min(abs(control_y), 15))  # 限制速度范围
            move_time = max(0.2, min(abs(err_y) / 300, 0.8))  # 限制时间范围
            adjust_y(math.copysign(move_speed, control_y), move_time)
            print("Y轴PID精细调整")
        else:
            # X轴调整，使用PID控制
            dog.gait_type("trot")
            # 将PID输出转换为实际的移动参数
            move_speed = max(8, min(abs(control_x), 12))  # 限制速度范围
            move_time = max(0.5, min(abs(err_x) / 200, 1.2))  # 限制时间范围
            adjust_x(math.copysign(move_speed, control_x), move_time)
            print("X轴PID调整")

    # # 角度纠偏
    # dog.gait_type("trot")
    # adjust_to_yaw(origin_yaw, 10)

    time.sleep(0.3)
    return False


def adjust_place(m_angle, m_x, m_y, opposite_yaw, des_x=1200, des_y=980):
    """
    使用PID控制算法调整机器人位置到目标放置点
    Args:
        m_angle: 物体角度
        m_x: 当前X坐标
        m_y: 当前Y坐标
        opposite_yaw: 目标朝向角度
        des_x: 目标X坐标
        des_y: 目标Y坐标
    Returns:
        True表示到达目标位置并完成放置，False表示还需要继续调整
    """
    origin_yaw = opposite_yaw

    # 计算位置误差
    err_x = des_x - m_x
    err_y = des_y - m_y

    # 使用PID控制器计算控制输出
    control_x = pid_x.update(err_x)
    control_y = pid_y.update(err_y)

    # Y轴方向的控制（左右移动）
    if abs(err_y) > 500:
        # 大范围偏移，使用传统控制方式
        adjust_y(math.copysign(12, err_y), max(0.3, min(abs(err_y) / 200, 1.5)))
    else:
        # X轴方向的控制（前后移动）
        if abs(err_x) < 70:
            if abs(err_y) < 40:
                # 到达目标位置，执行放置
                place_two()
                return True
            else:
                # 细微Y轴调整，使用PID控制
                dog.gait_type("slow_trot")
                move_speed = max(8, min(abs(control_y), 15))
                move_time = max(0.2, min(abs(err_y) / 300, 0.8))
                adjust_y(math.copysign(move_speed, control_y), move_time)
        else:
            # X轴调整，使用PID控制
            dog.gait_type("trot")
            move_speed = max(8, min(abs(control_x), 12))
            move_time = max(0.5, min(abs(err_x) / 200, 1.2))
            adjust_x(math.copysign(move_speed, control_x), move_time)

    # 角度纠偏
    dog.gait_type("trot")
    adjust_to_yaw(origin_yaw, 10)

    time.sleep(0.3)
    return False


def seek_pick(opposite_yaw, color='blue'):
    """
    寻找并拾取指定颜色的物体
    Args:
        opposite_yaw: 目标朝向角度
        color: 要寻找的颜色 ('blue', 'green', 'red')
    """
    # 重置PID控制器
    pid_x.reset()
    pid_y.reset()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # 根据颜色设置HSV范围
    if color == 'blue':
        min_blue = [90, 50, 0]
        max_blue = [110, 255, 255]
    elif color == 'green':
        min_blue = [65, 150, 0]
        max_blue = [80, 255, 255]

    m_angle, m_x, m_y = 0, 0, 0
    count = 0
    COUNT_MAX = 20  # 平均化计算的帧数
    ready_for_grasp()  # 准备抓取姿势
    j = 0

    while 1:
        # 获取一帧图像
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
        else:
            frame = cv2.resize(frame, (1920, 1440))

        # 根据颜色类型进行图像过滤
        if color == 'red':
            frame_filter = filter_img_red(frame, color_ranges_red)
        else:
            frame_filter = filter_img(frame, [min_blue, max_blue])

        # 检测轮廓
        counters, frame = detect_contours(frame_filter)

        # 检测块状物体
        flag, length, width, angle, s_x, s_y, frame = detect_block(counters, frame_filter)
        j += 1

        if flag:
            count += 1
            # 使用滑动平均来平滑检测结果
            m_angle = (count - 1) / count * m_angle + angle / count
            m_x = (count - 1) / count * m_x + s_x / count
            m_y = (count - 1) / count * m_y + s_y / count

        # 当累积足够的检测结果后进行位置调整
        if count == COUNT_MAX:
            count = 0
            res = adjust(m_angle, m_x, m_y, opposite_yaw=opposite_yaw)
            if res:
                break

    cap.release()
    cv2.destroyAllWindows()


def seek_place(opposite_yaw, color='blue'):
    """
    寻找并放置物体到指定颜色区域
    Args:
        opposite_yaw: 目标朝向角度
        color: 目标颜色区域 ('blue', 'green', 'red')
    """
    # 重置PID控制器
    pid_x.reset()
    pid_y.reset()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # 根据颜色设置HSV范围
    if color == 'blue':
        min_blue = [90, 50, 0]
        max_blue = [110, 255, 255]
    elif color == 'green':
        min_blue = [65, 150, 0]
        max_blue = [80, 255, 255]

    m_angle, m_x, m_y = 0, 0, 0
    count = 0
    COUNT_MAX = 20
    ready_for_grasp()
    j = 0

    while 1:
        # 获取一帧图像
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame")
        else:
            frame = cv2.resize(frame, (1920, 1440))

        # 根据颜色类型进行图像过滤
        if color == 'red':
            frame_filter = filter_img_red(frame, color_ranges_red)
        else:
            frame_filter = filter_img(frame, [min_blue, max_blue])

        # 保存调试图像（每5帧保存一次）
        if j % 5 == 0:
            print("saved>>>>>")
            string = 'pick_log/' + str(j) + '.jpg'
            cv2.imwrite(string, frame)

        # 检测轮廓
        counters, frame = detect_contours(frame_filter)

        # 检测块状物体
        flag, length, width, angle, s_x, s_y, frame = detect_block(counters, frame_filter)

        # 保存处理后的图像
        if j % 5 == 0:
            print("saved>>>>>")
            string = 'pick_log/' + str(j) + '.jpg'
            cv2.imwrite(string, frame)
        j += 1

        if flag:
            count += 1
            # 使用滑动平均来平滑检测结果
            m_angle = (count - 1) / count * m_angle + angle / count
            m_x = (count - 1) / count * m_x + s_x / count
            m_y = (count - 1) / count * m_y + s_y / count

        # 当累积足够的检测结果后进行位置调整
        if count == COUNT_MAX:
            print("我的位置", m_x, m_y)
            count = 0
            res = adjust_place(m_angle, m_x, m_y, opposite_yaw=opposite_yaw)
            if res:
                break

    cap.release()
    cv2.destroyAllWindows()


def run(opposite_yaw, color='blue', is_place=False):
    """
    主运行函数：根据参数决定是执行拾取还是放置任务
    Args:
        opposite_yaw: 目标朝向角度
        color: 目标颜色
        is_place: True表示放置任务，False表示拾取任务
    """
    origin_yaw = opposite_yaw

    if is_place:
        seek_place(opposite_yaw, color)
    else:
        while True:
            seek_pick(origin_yaw, color)

            # 读取51号舵机角度【-65， 65】，-65是完全打开
            c = dog.read_motor()
            print(c[12])

            # 检查是否成功抓取物体
            if c[12] > 0 and c[12] < 50:
                break

            # 如果没有抓取到物体，后退一点重新尝试
            dog.gait_type("trot")
            adjust_x(-10, 1)


def grasp():
    """
    执行抓取动作的函数
    """
    dog.claw(0)  # 张开爪子
    time.sleep(0.5)
    dog.translation("x", 20)  # 向前伸展
    dog.motor(52, -55)  # 调整关节角度
    time.sleep(0.5)
    dog.translation("z", 60)  # 向下伸展
    time.sleep(0.5)
    dog.motor(53, 80)  # 调整关节角度
    dog.attitude("p", 20)  # 调整俯仰角度
    time.sleep(2)
    dog.claw(255)  # 闭合爪子抓取
    time.sleep(2)
    dog.reset()  # 回到初始状态
    time.sleep(1)
    dog.motor(52, -55)  # 保持抓取状态
    time.sleep(1)


if __name__ == '__main__':
    # 获取当前朝向作为基准朝向
    for i in range(5):
        oppozite_yaw = dog.read_yaw()
    left_yaw = oppozite_yaw + 90

    # 执行红色物体的拾取任务
    run(oppozite_yaw, color='red', is_place=False)
