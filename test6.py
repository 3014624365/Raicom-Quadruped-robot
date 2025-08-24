import cv2
import time
import numpy as np
from xgolib import XGO

# 实例化dog
dog = XGO("xgolite")
dog.attitude('p', 40)

# 定义颜色范围
color_dist = {
    'red': {'Lower1': np.array([0, 60, 60]), 'Upper1': np.array([6, 255, 255]),
            'Lower2': np.array([170, 60, 60]), 'Upper2': np.array([180, 255, 255])},
    'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
    'green': {'Lower': np.array([35, 100, 100]), 'Upper': np.array([85, 255, 255])},
    'yellow': {'Lower': np.array([17, 100, 100]), 'Upper': np.array([30, 255, 255])},
}


def apply_morphology(mask, kernel_size=(7, 7), iterations=2):
    """
    应用形态学操作：腐蚀 + 膨胀
    用于消除噪声并增强轮廓
    """
    kernel = np.ones(kernel_size, np.uint8)
    mask_erode = cv2.erode(mask, kernel, iterations=iterations)
    mask_dilate = cv2.dilate(mask_erode, kernel, iterations=iterations)
    return mask_dilate


def get_rectangle_rotation_angle(image, min_area=500):
    """
    获取掩膜图像中最大矩形轮廓的旋转角度和中心位置

    Args:
        image: 输入的二值化图像或掩膜图像
        min_area: 轮廓的最小面积阈值，用于过滤小轮廓

    Returns:
        angle: 检测到的矩形旋转角度（-90到90度范围内，0度表示长边竖直）
        center: 矩形的中心点坐标 (x, y)
        rect_info: 包含矩形信息的字典
        vis_image: 可视化后的图像
    """
    # 如果输入是彩色图像，转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 确保图像是二值图像
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    # 阈值化处理，确保是二值图像
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 形态学操作增强轮廓
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=3)
    binary = cv2.erode(binary, kernel, iterations=3)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建可视化图像
    if len(image.shape) == 3:
        vis_image = image.copy()
    else:
        vis_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # 初始化返回值
    angle = None
    center = (0, 0)
    largest_area = 0
    largest_contour = None
    rect_info = None

    # 遍历所有轮廓，找到面积最大的轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area > largest_area:
            largest_area = area
            largest_contour = contour

    # 如果找到了有效轮廓
    if largest_contour is not None:
        # 计算最小外接矩形
        rect = cv2.minAreaRect(largest_contour)

        # 获取矩形中心点、宽高和旋转角度
        center = rect[0]  # 中心点坐标 (x, y)
        size = rect[1]  # 宽高 (width, height)
        cv_angle = rect[2]  # 旋转角度 (OpenCV返回值)

        # 角度处理：将OpenCV的角度转换为我们需要的角度
        # OpenCV返回的角度范围是-90到0度
        # 我们需要转换为机器人偏航角度：-90到90度
        if size[0] > size[1]:  # 宽 > 高
            angle = cv_angle
        else:  # 高 > 宽
            angle = cv_angle + 90

        # 确保角度在-90到90范围内
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        # 存储矩形信息
        rect_info = {
            'area': largest_area,
            'rect': rect,
            'size': size
        }

        # 绘制最小外接矩形
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(vis_image, [box], 0, (0, 255, 0), 3)

        # 绘制中心点
        cv2.circle(vis_image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

        # 添加角度文本
        if angle is not None:
            cv2.putText(vis_image, f"Angle: {angle:.2f} deg",
                        (int(center[0]) - 100, int(center[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return angle, center, rect_info, vis_image


def get_color_mask(hsv_filtered, color_name):
    """
    根据颜色名称获取对应的掩码

    Args:
        hsv_filtered: HSV格式的图像
        color_name: 颜色名称 ('red', 'blue', 'green', 'yellow')

    Returns:
        mask: 二值掩码图像
    """
    if color_name == 'red':
        # 红色需要两个范围（因为红色在HSV色相环的两端）
        mask1 = cv2.inRange(hsv_filtered, color_dist['red']['Lower1'], color_dist['red']['Upper1'])
        mask2 = cv2.inRange(hsv_filtered, color_dist['red']['Lower2'], color_dist['red']['Upper2'])
        return cv2.bitwise_or(mask1, mask2)
    else:
        return cv2.inRange(hsv_filtered, color_dist[color_name]['Lower'], color_dist[color_name]['Upper'])


def zhuaqu(angle=None):
    """
    抓取函数，可以根据检测到的角度调整抓取策略

    Args:
        angle: 检测到的矩形旋转角度
    """
    print(f"执行抓取 - 角度: {angle:.2f}°" if angle is not None else "执行抓取 - 未检测到角度")

    # 根据角度决定抓取方式
    if angle is not None:
        if abs(angle) < 15:  # 接近正向
            print("物体接近正向，使用标准抓取方式")
        elif abs(angle) > 75:  # 接近侧向
            print("物体接近侧向，使用侧向抓取方式")
        else:
            print(f"物体倾斜 {angle:.1f}°，调整抓取策略")

    # 执行抓取动作序列
    dog.attitude('p', 40)  # 设置机身俯仰角
    dog.claw(0)  # 张开机械爪
    dog.arm(0, 100)  # 机械臂末端位于基座正上方100mm处
    dog.arm(90, 90)  # 机械臂大臂垂直于身体，小臂水平于身体
    dog.arm(100, 0)  # 机械臂末端在摄像头正前方
    dog.arm(100, -60)  # 机械臂末端下探抓取
    time.sleep(2)  # 等待2秒

    dog.claw(200)  # 夹住物体
    time.sleep(1)  # 等待1秒

    dog.arm(0, 90)  # 收回机械臂
    dog.attitude('p', 0)  # 恢复站立
    time.sleep(1)  # 等待1秒


def reset_state():
    """重置状态变量"""
    return {
        'phase': "yaw_correction",  # 当前阶段：偏航修正 -> x_adjustment -> y_adjustment
        'stable_count': 0,  # 稳定计数
        'found_contour': False,  # 是否找到轮廓
        'last_angle': None,  # 最后检测到的角度
        'last_center': None,  # 最后检测到的中心点
        'rect_info': None,  # 矩形信息
    }


def yaw_correction_phase(frame, angle, center, state):
    """
    偏航修正阶段：使机器狗正对长方体

    Args:
        frame: 当前视频帧
        angle: 检测到的角度
        center: 矩形中心点
        state: 状态字典

    Returns:
        bool: 是否完成偏航修正
    """
    # 偏航修正的角度阈值
    YAW_ANGLE_THRESHOLD = 10  # 角度小于10度认为已对齐
    STABLE_REQUIREMENT = 3  # 需要连续3次稳定才认为完成

    cv2.putText(frame, "Phase: YAW CORRECTION", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if angle is not None:
        cv2.putText(frame, f"Angle: {angle:.2f} deg", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Stable: {state['stable_count']}/{STABLE_REQUIREMENT}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 检查角度是否在阈值范围内
        if abs(angle) <= YAW_ANGLE_THRESHOLD:
            dog.stop()
            state['stable_count'] += 1
            print(f"[偏航修正] 角度稳定 {state['stable_count']}/{STABLE_REQUIREMENT}，角度: {angle:.2f}°")

            if state['stable_count'] >= STABLE_REQUIREMENT:
                print("===== 偏航修正完成，开始X轴调整 =====")
                state['phase'] = "x_adjustment"
                state['stable_count'] = 0
                return True
        else:
            # 需要偏航修正
            state['stable_count'] = 0
            yaw_speed = 8  # 偏航速度

            if angle > 0:
                dog.move("yaw", -yaw_speed)  # 逆时针转
                print(f"[偏航修正] 逆时针转动，角度: {angle:.2f}°")
            else:
                dog.move("yaw", yaw_speed)  # 顺时针转
                print(f"[偏航修正] 顺时针转动，角度: {angle:.2f}°")
    else:
        # 没有检测到角度，停止移动
        dog.stop()
        print("[偏航修正] 未检测到有效角度")

    return False


def x_axis_adjustment_phase(frame, center, state):
    """
    X轴调整阶段：前进后退调整

    Args:
        frame: 当前视频帧
        center: 矩形中心点
        state: 状态字典

    Returns:
        bool: 是否完成X轴调整
    """
    # X轴调整的位置阈值
    X_TARGET_Y = 400  # 目标Y坐标（图像下方，靠近机器狗）
    X_THRESHOLD = 30  # Y坐标误差阈值
    STABLE_REQUIREMENT = 3  # 稳定要求

    cv2.putText(frame, "Phase: X-AXIS ADJUSTMENT", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # 图像中心线
    img_center_x = frame.shape[1] // 2
    cv2.line(frame, (0, X_TARGET_Y), (frame.shape[1], X_TARGET_Y), (0, 255, 255), 2)

    if center is not None:
        center_x, center_y = center
        y_diff = center_y - X_TARGET_Y

        cv2.putText(frame, f"Y Diff: {y_diff:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Stable: {state['stable_count']}/{STABLE_REQUIREMENT}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 检查Y轴位置是否在阈值范围内
        if abs(y_diff) <= X_THRESHOLD:
            dog.stop()
            state['stable_count'] += 1
            print(f"[X轴调整] 位置稳定 {state['stable_count']}/{STABLE_REQUIREMENT}，Y差值: {y_diff:.1f}")

            if state['stable_count'] >= STABLE_REQUIREMENT:
                print("===== X轴调整完成，开始Y轴调整 =====")
                state['phase'] = "y_adjustment"
                state['stable_count'] = 0
                return True
        else:
            # 需要X轴调整
            state['stable_count'] = 0
            x_speed = 4

            if y_diff > 0:  # 物体在目标线下方，需要前进
                dog.move("x", x_speed)
                print(f"[X轴调整] 前进，Y差值: {y_diff:.1f}")
            else:  # 物体在目标线上方，需要后退
                dog.move("x", -x_speed)
                print(f"[X轴调整] 后退，Y差值: {y_diff:.1f}")
    else:
        dog.stop()
        print("[X轴调整] 未检测到有效中心点")

    return False


def y_axis_adjustment_phase(frame, center, state):
    """
    Y轴调整阶段：左右调整

    Args:
        frame: 当前视频帧
        center: 矩形中心点
        state: 状态字典

    Returns:
        bool: 是否完成Y轴调整（可以执行抓取）
    """
    # Y轴调整的位置阈值
    Y_THRESHOLD = 20  # X坐标误差阈值
    STABLE_REQUIREMENT = 3  # 稳定要求

    cv2.putText(frame, "Phase: Y-AXIS ADJUSTMENT", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # 图像中心
    img_center_x = frame.shape[1] // 2
    cv2.line(frame, (img_center_x, 0), (img_center_x, frame.shape[0]), (255, 0, 255), 2)

    if center is not None:
        center_x, center_y = center
        x_diff = center_x - img_center_x

        cv2.putText(frame, f"X Diff: {x_diff:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Stable: {state['stable_count']}/{STABLE_REQUIREMENT}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # 检查X轴位置是否在阈值范围内
        if abs(x_diff) <= Y_THRESHOLD:
            dog.stop()
            state['stable_count'] += 1
            print(f"[Y轴调整] 位置稳定 {state['stable_count']}/{STABLE_REQUIREMENT}，X差值: {x_diff:.1f}")

            if state['stable_count'] >= STABLE_REQUIREMENT:
                print("===== Y轴调整完成，执行抓取 =====")
                return True
        else:
            # 需要Y轴调整
            state['stable_count'] = 0
            y_speed = 4

            if x_diff > 0:  # 物体在中心线右侧，需要右移
                dog.move("y", -y_speed)
                print(f"[Y轴调整] 右移，X差值: {x_diff:.1f}")
            else:  # 物体在中心线左侧，需要左移
                dog.move("y", y_speed)
                print(f"[Y轴调整] 左移，X差值: {x_diff:.1f}")
    else:
        dog.stop()
        print("[Y轴调整] 未检测到有效中心点")

    return False


def process_frame_with_phases(frame, target_color, state):
    """
    处理每一帧图像，执行三阶段调整流程

    Args:
        frame: 当前视频帧
        target_color: 目标颜色
        state: 状态字典

    Returns:
        bool: 是否需要执行抓取
    """
    # 图像预处理
    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)
    hsv_filtered = cv2.bilateralFilter(cv2.medianBlur(hsv, 5), 9, 75, 75)

    # 获取目标颜色的掩码
    mask = get_color_mask(hsv_filtered, target_color)
    mask = apply_morphology(mask)

    # 检测角度和中心点
    angle, center, rect_info, vis_image = get_rectangle_rotation_angle(mask, min_area=500)

    # 更新状态
    state['last_angle'] = angle
    state['last_center'] = center
    state['rect_info'] = rect_info
    state['found_contour'] = (center is not None)

    # 绘制检测结果
    if center is not None:
        cv2.circle(frame, (int(center[0]), int(center[1])), 8, (0, 255, 255), -1)

        # 绘制轮廓
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for c in cnts:
            if cv2.contourArea(c) >= 500:
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

    # 根据当前阶段执行相应的调整
    if state['phase'] == "yaw_correction":
        # 阶段1：偏航修正
        if yaw_correction_phase(frame, angle, center, state):
            pass  # 进入下一阶段

    elif state['phase'] == "x_adjustment":
        # 阶段2：X轴调整
        if x_axis_adjustment_phase(frame, center, state):
            pass  # 进入下一阶段

    elif state['phase'] == "y_adjustment":
        # 阶段3：Y轴调整
        if y_axis_adjustment_phase(frame, center, state):
            return True  # 可以执行抓取

    # 如果没有找到目标，停止移动
    if not state['found_contour']:
        dog.stop()
        cv2.putText(frame, "NO TARGET FOUND", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return False


def check_grab_success(hsv_filtered, color_name, min_area=500):
    """
    检查抓取是否成功（视野内是否还有目标颜色）

    Args:
        hsv_filtered: 预处理后的HSV图像
        color_name: 目标颜色名称
        min_area: 最小面积阈值

    Returns:
        bool: True表示抓取成功（没有目标物体），False表示抓取失败
    """
    mask = get_color_mask(hsv_filtered, color_name)
    cnts = cv2.findContours(apply_morphology(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # 检查是否还有足够大的轮廓
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            return False  # 还有目标物体，抓取失败
    return True  # 没有目标物体，抓取成功


def run(target_color):
    """
    主运行函数：三阶段调整 + 抓取

    运行流程：
    1. 偏航修正：使机器狗正对长方体
    2. X轴调整：前后调整到合适距离
    3. Y轴调整：左右调整到中心位置
    4. 执行抓取
    5. 验证抓取结果

    Args:
        target_color: 要抓取的颜色，可选 'red', 'blue', 'green', 'yellow'
    """
    print(f"开始三阶段抓取流程：{target_color} 色物体")
    print("流程：偏航修正 → X轴调整 → Y轴调整 → 抓取")

    # 颜色对应的BGR值（用于绘制边界框）
    color_bgr = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'yellow': (0, 255, 255)
    }

    if target_color not in color_dist:
        print(f"不支持的颜色: {target_color}")
        return

    # 初始化摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
    dog.pace('low')

    # 初始化状态
    state = reset_state()
    grab_attempts = 0
    max_attempts = 3

    try:
        while cap.isOpened() and grab_attempts < max_attempts:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("无法读取视频帧")
                break

            # 显示尝试次数
            cv2.putText(frame, f"Attempt: {grab_attempts + 1}/{max_attempts}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # 处理当前帧，执行三阶段调整
            grab_ready = process_frame_with_phases(frame, target_color, state)

            # 如果完成了所有调整阶段，执行抓取
            if grab_ready:
                print("===== 开始执行抓取 =====")
                dog.rider_led(3, [0, 0, 255])  # 红灯表示抓取中

                # 执行抓取，传入最后检测到的角度
                zhuaqu(state['last_angle'])

                dog.rider_led(3, [0, 0, 0])  # 灭灯
                grab_attempts += 1
                print(f"第 {grab_attempts} 次抓取尝试完成")

                # 等待一段时间让物体稳定
                time.sleep(2)

                # 检查抓取是否成功
                ret, check_frame = cap.read()
                if ret and check_frame is not None:
                    gs_check_frame = cv2.GaussianBlur(check_frame, (5, 5), 0)
                    hsv_check = cv2.cvtColor(gs_check_frame, cv2.COLOR_BGR2HSV)
                    hsv_check_filtered = cv2.bilateralFilter(cv2.medianBlur(hsv_check, 5), 9, 75, 75)

                    if check_grab_success(hsv_check_filtered, target_color):
                        print(f"===== {target_color} 色物体抓取成功！=====")
                        dog.rider_led(3, [0, 255, 0])  # 绿灯表示成功
                        time.sleep(3)
                        dog.rider_led(3, [0, 0, 0])  # 灭灯
                        break
                    else:
                        print(f"===== 第 {grab_attempts} 次抓取失败，重新开始流程 =====")
                        dog.rider_led(3, [255, 0, 0])  # 红灯表示失败
                        time.sleep(2)
                        dog.rider_led(3, [0, 0, 0])  # 灭灯

                        # 重置状态，重新开始三阶段流程
                        state = reset_state()

            # 显示图像
            cv2.imshow('camera', frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出程序")
                break
            elif key == ord('r'):  # 'r'键重置状态
                print("重置调整状态")
                state = reset_state()
            elif key == ord(' '):  # 空格键切换颜色
                colors = list(color_dist.keys())
                current_idx = colors.index(target_color)
                target_color = colors[(current_idx + 1) % len(colors)]
                print(f"切换目标颜色为: {target_color}")
                state = reset_state()

        if grab_attempts >= max_attempts:
            print(f"===== 达到最大尝试次数 ({max_attempts})，停止抓取 =====")

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        dog.stop()
        print("程序结束，资源已清理")


# 使用示例
if __name__ == "__main__":
    # 运行抓取程序，目标颜色为红色
    run(target_color='red')
