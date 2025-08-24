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
}

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)

# 调整阈值参数
horizontal_threshold = 20  # 水平调整阈值
vertical_threshold_low = 130  # 竖直调整下限阈值
vertical_threshold_high = 140  # 竖直调整上限阈值
dog.pace('low')

# 调整状态变量
adjustment_phase = "horizontal"  # 初始阶段：horizontal(水平) -> vertical(竖直) -> horizontal(水平)
horizontal_stable_count = 0  # 水平稳定计数
vertical_stable_count = 0  # 竖直稳定计数
STABLE_REQUIREMENT = 5  # 连续稳定次数要求
found_contour = False  # 是否检测到有效轮廓
horizontal_completed = False  # 水平调整是否已完成（第一次）
vertical_completed = False  # 竖直调整是否已完成
second_horizontal_completed = False  # 第二次水平调整是否已完成

def apply_morphology(mask, kernel_size=(7, 7), iterations=2):
    """应用形态学操作：腐蚀 + 膨胀"""
    kernel = np.ones(kernel_size, np.uint8)
    mask_erode = cv2.erode(mask, kernel, iterations=iterations)
    mask_dilate = cv2.dilate(mask_erode, kernel, iterations=iterations)
    return mask_dilate

# 定义抓取函数
def zhuaqu():
    # 设置机身俯仰角、偏航角、滚转角
    dog.attitude('p', 40)
    
    # 设置机械爪夹角
    dog.claw(0)
    time.sleep(1)
    # 设置机械臂位置
    dog.arm(0, 100)  # 机械臂末端位于基座正上方100mm处
    dog.arm(90, 90)  # 机械臂大臂垂直于身体，小臂水平于身体
    dog.arm(100, 0)  # 机械臂末端在摄像头正前方
    dog.arm(100, -50)  # 机械臂末端下探抓取
    time.sleep(2)  # 等待2秒

    # 夹住物体
    dog.claw(255)

    time.sleep(2)
     # 等待1秒

    # 收回机械臂
    dog.arm(0, 90)
    
    # 恢复站立
    dog.attitude('p', 0)
    
    time.sleep(1)  # 等待1秒
    print("===== 抓取完成，程序退出 =====")
    exit()  # 抓取完成后退出程序

def draw_contours(frame, cnts, color):
    """根据颜色轮廓绘制边界框并计算中心点差值，水平调整一次完成后不再调整"""
    global adjustment_phase, horizontal_stable_count, vertical_stable_count, found_contour, horizontal_completed, vertical_completed, second_horizontal_completed
    found_contour = False  # 重置为未找到
    # 图像的中心点
    img_center_x = frame.shape[1] // 2
    img_center_y = frame.shape[0] // 2
    cv2.circle(frame, (img_center_x, img_center_y), 5, (255, 255, 255), -1)  # 图像中心点
    
    # 在画面上显示当前调整阶段
    cv2.putText(frame, f"Phase: {adjustment_phase}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Horizontal Completed: {horizontal_completed}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Vertical Completed: {vertical_completed}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Second Horizontal Completed: {second_horizontal_completed}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    for c in cnts:
        if cv2.contourArea(c) < 500:
            continue
        
        found_contour = True  # 标记找到有效轮廓
        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(c)
        rect_center_x = x + w // 2
        rect_center_y = y + h // 2

        # 绘制轮廓和外接矩形
        cv2.drawContours(frame, [c], -1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (rect_center_x, rect_center_y), 5, (0, 255, 255), -1)  # 矩形中心点

        # 计算水平和竖直差值
        horizontal_diff = rect_center_x - img_center_x
        vertical_diff = rect_center_y - img_center_y

        # 阶段1：第一次水平调整
        if adjustment_phase == "horizontal" and not horizontal_completed:
            # 显示水平调整信息
            cv2.putText(frame, f"Horiz Diff: {horizontal_diff}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Stable: {horizontal_stable_count}/{STABLE_REQUIREMENT}", (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if abs(horizontal_diff) <= horizontal_threshold:
                # 水平已在阈值范围内
                dog.stop()
                horizontal_stable_count += 1
                print(f"[水平调整] 稳定 {horizontal_stable_count}/{STABLE_REQUIREMENT}，差值: {horizontal_diff}")
                
                # 连续稳定达到要求，切换到竖直调整阶段
                if horizontal_stable_count >= STABLE_REQUIREMENT:
                    print("===== 水平调整完成，开始竖直调整 =====")
                    adjustment_phase = "vertical"
                    horizontal_completed = True  # 标记第一次水平调整已完成
                    horizontal_stable_count = 0  # 重置计数
            else:
                # 水平未到位，继续调整
                horizontal_stable_count = 0  # 重置稳定计数
                if horizontal_diff > 0:
                    dog.move("y", -3.5)  # 向右调整
                    print(f"[水平调整] 向右移动，差值: {horizontal_diff}")
                else:
                    dog.move("y", 3.5)  # 向左调整
                    print(f"[水平调整] 向左移动，差值: {horizontal_diff}")

        # 阶段2：竖直调整
        elif adjustment_phase == "vertical" and horizontal_completed and not vertical_completed:
            # 显示竖直调整信息
            cv2.putText(frame, f"Vert Diff: {vertical_diff}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Stable: {vertical_stable_count}/{STABLE_REQUIREMENT}", (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 检查竖直是否在目标范围内
            if vertical_threshold_low <= vertical_diff <= vertical_threshold_high:
                # 竖直已在阈值范围内
                dog.stop()
                vertical_stable_count += 1
                print(f"[竖直调整] 稳定 {vertical_stable_count}/{STABLE_REQUIREMENT}，差值: {vertical_diff}")
                
                # 连续稳定达到要求，等待摄像头稳定后切换到第二次水平调整阶段
                if vertical_stable_count >= STABLE_REQUIREMENT:
                    print("===== 竖直调整完成，等待摄像头稳定 =====")
                    time.sleep(1.5)  # 等待1.5秒让摄像头画面稳定
                    print("===== 开始第二次水平调整 =====")
                    adjustment_phase = "second_horizontal"
                    vertical_completed = True  # 标记竖直调整已完成
                    vertical_stable_count = 0  # 重置计数
                    horizontal_stable_count = 0  # 重置水平稳定计数
            else:
                # 竖直未到位，继续调整
                vertical_stable_count = 0  # 重置稳定计数
                if vertical_diff < vertical_threshold_low:
                    dog.move("x", 2)  # 向前调整
                    print(f"[竖直调整] 向前移动，差值: {vertical_diff}")
                else:
                    dog.move("x", -2)  # 向后调整
                    print(f"[竖直调整] 向后移动，差值: {vertical_diff}")

        # 阶段3：第二次水平调整
        elif adjustment_phase == "second_horizontal" and vertical_completed and not second_horizontal_completed:
            # 显示第二次水平调整信息
            cv2.putText(frame, f"Horiz Diff (Second): {horizontal_diff}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Stable: {horizontal_stable_count}/{STABLE_REQUIREMENT}", (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if abs(horizontal_diff) <= horizontal_threshold:
                # 水平已在阈值范围内
                dog.stop()
                horizontal_stable_count += 1
                print(f"[第二次水平调整] 稳定 {horizontal_stable_count}/{STABLE_REQUIREMENT}，差值: {horizontal_diff}")
                
                # 连续稳定达到要求，执行抓取
                if horizontal_stable_count >= STABLE_REQUIREMENT:
                    print("===== 第二次水平调整完成，执行抓取 =====")
                    dog.rider_led(3, [0,0,255])
                    zhuaqu()  # 执行抓取函数
                    dog.rider_led(3, [0,0,0])  # 灭灯
                    second_horizontal_completed = True  # 标记第二次水平调整已完成
                    horizontal_stable_count = 0  # 重置计数
            else:
                # 水平未到位，继续调整
                horizontal_stable_count = 0  # 重置稳定计数
                if horizontal_diff > 0:
                    dog.move("y", -1.5)  # 向右调整
                    print(f"[第二次水平调整] 向右移动，差值: {horizontal_diff}")
                else:
                    dog.move("y", 1.5)  # 向左调整
                    print(f"[第二次水平调整] 向左移动，差值: {horizontal_diff}")

    # 如果没有找到有效轮廓，停止移动
    if not found_contour:
        dog.stop()
        print("未检测到有效轮廓，停止移动")

def pick_cuboid()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)
        hsv_filtered = cv2.bilateralFilter(cv2.medianBlur(hsv, 5), 9, 75, 75)

        # 蓝色掩码（当前使用的识别颜色）
        mask_blue = cv2.inRange(hsv_filtered, color_dist['blue']['Lower'], color_dist['blue']['Upper'])
        cnts_blue = cv2.findContours(apply_morphology(mask_blue), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        draw_contours(frame, cnts_blue, (255, 0, 0))  # 蓝色边界框

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
