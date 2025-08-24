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
    获取掩膜图像中最大矩形轮廓的旋转角度
    不进行平滑处理，直接返回检测到的角度

    Args:
        image: 输入的二值化图像或掩膜图像
        min_area: 轮廓的最小面积阈值，用于过滤小轮廓

    Returns:
        angle: 检测到的矩形旋转角度（-90到90度范围内）
        center: 矩形的中心点坐标 (x, y)
        area: 检测到的矩形面积
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
    area = 0
    largest_area = 0
    largest_contour = None

    # 遍历所有轮廓，找到面积最大的轮廓
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > min_area and contour_area > largest_area:
            largest_area = contour_area
            largest_contour = contour

    # 如果找到了有效轮廓
    if largest_contour is not None:
        # 计算最小外接矩形
        rect = cv2.minAreaRect(largest_contour)

        # 获取矩形中心点、宽高和旋转角度
        center = rect[0]  # 中心点坐标 (x, y)
        size = rect[1]  # 宽高 (width, height)
        cv_angle = rect[2]  # OpenCV返回的旋转角度

        area = largest_area

        # 角度处理：将OpenCV的角度转换为我们需要的偏航角度
        # OpenCV返回的角度范围是-90到0度
        if size[0] > size[1]:  # 宽 > 高（横向矩形）
            angle = cv_angle
        else:  # 高 > 宽（纵向矩形）
            angle = cv_angle + 90

        # 确保角度在-90到90范围内
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        # 绘制最小外接矩形
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(vis_image, [box], 0, (0, 255, 0), 3)

        # 绘制中心点
        cv2.circle(vis_image, (int(center[0]), int(center[1])), 8, (0, 0, 255), -1)

        # 添加角度文本
        cv2.putText(vis_image, f"Angle: {angle:.1f} deg",
                    (int(center[0]) - 100, int(center[1]) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 添加面积文本
        cv2.putText(vis_image, f"Area: {int(area)}",
                    (int(center[0]) - 100, int(center[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 绘制角度指示线（显示物体朝向）
        line_length = 80
        end_x = int(center[0] + line_length * np.sin(np.radians(angle)))
        end_y = int(center[1] - line_length * np.cos(np.radians(angle)))
        cv2.line(vis_image, (int(center[0]), int(center[1])), (end_x, end_y), (0, 255, 255), 3)

        # 绘制垂直参考线（0度方向）
        ref_end_y = int(center[1] - 60)
        cv2.line(vis_image, (int(center[0]), int(center[1])), (int(center[0]), ref_end_y), (255, 255, 255), 2)

    return angle, center, area, vis_image


def get_color_mask(hsv_filtered, color_name):
    """
    根据颜色名称获取对应的掩码

    Args:
        hsv_filtered: HSV格式的图像
        color_name: 颜色名称

    Returns:
        mask: 二值掩码图像
    """
    if color_name == 'red':
        # 红色需要两个范围
        mask1 = cv2.inRange(hsv_filtered, color_dist['red']['Lower1'], color_dist['red']['Upper1'])
        mask2 = cv2.inRange(hsv_filtered, color_dist['red']['Lower2'], color_dist['red']['Upper2'])
        return cv2.bitwise_or(mask1, mask2)
    else:
        return cv2.inRange(hsv_filtered, color_dist[color_name]['Lower'], color_dist[color_name]['Upper'])


def yaw_correction_test(target_color='red', angle_threshold=10, yaw_speed=45):
    """
    偏航修正测试函数
    只进行角度检测和偏航调整，不进行其他动作

    Args:
        target_color: 目标检测颜色
        angle_threshold: 角度阈值（度），小于此值认为已对齐
        yaw_speed: 偏航调整速度
    """
    print(f"=== 偏航修正测试开始 ===")
    print(f"目标颜色: {target_color}")
    print(f"角度阈值: ±{angle_threshold}°")
    print(f"偏航速度: {yaw_speed}")
    print(f"控制说明: 'q'退出, 'r'重置, 空格键切换颜色")
    print("-" * 40)

    # 初始化摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    cv2.namedWindow('Yaw Correction Test', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Mask', cv2.WINDOW_AUTOSIZE)
    dog.pace('low')

    # 测试统计
    frame_count = 0
    detection_count = 0
    correction_count = 0
    aligned_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("警告：无法读取视频帧")
                break

            frame_count += 1

            # 图像预处理
            gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)
            hsv_filtered = cv2.bilateralFilter(cv2.medianBlur(hsv, 5), 9, 75, 75)

            # 获取目标颜色的掩码
            mask = get_color_mask(hsv_filtered, target_color)
            mask = apply_morphology(mask)

            # 检测角度
            angle, center, area, vis_mask = get_rectangle_rotation_angle(mask, min_area=500)

            # 在原图上绘制信息
            cv2.putText(frame, "YAW CORRECTION TEST", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Target: {target_color.upper()}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Threshold: +/-{angle_threshold} deg", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 处理检测结果
            if angle is not None:
                detection_count += 1

                # 在原图上绘制检测信息
                cv2.putText(frame, f"Angle: {angle:.1f} deg", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Center: ({center[0]:.0f}, {center[1]:.0f})", (10, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {int(area)}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 绘制中心点和角度信息
                cv2.circle(frame, (int(center[0]), int(center[1])), 10, (0, 255, 255), -1)

                # 绘制角度指示线
                line_length = 100
                end_x = int(center[0] + line_length * np.sin(np.radians(angle)))
                end_y = int(center[1] - line_length * np.cos(np.radians(angle)))
                cv2.line(frame, (int(center[0]), int(center[1])), (end_x, end_y), (0, 255, 255), 3)

                # 偏航修正逻辑（不进行平滑处理，直接使用检测到的角度）
                if abs(angle) <= angle_threshold:
                    # 角度在阈值内，已对齐
                    dog.stop()
                    aligned_count += 1
                    status = "ALIGNED"
                    status_color = (0, 255, 0)
                    print(f"[帧{frame_count:4d}] 已对齐 - 角度: {angle:6.1f}°")
                else:
                    # 需要偏航修正
                    correction_count += 1
                    if angle > 0:
                        dog.turn(yaw_speed)  # 逆时针转
                        time.sleep(angle/yaw_speed)
                        dog.turn(0)
                        status = f"TURN LEFT (CCW)"
                        status_color = (0, 0, 255)
                        print(f"[帧{frame_count:4d}] 逆时针转 - 角度: {angle:6.1f}° -> 速度: -{yaw_speed}")
                    else:
                        dog.turn(-yaw_speed)  # 逆时针转
                        time.sleep(angle / yaw_speed)
                        dog.turn(0)
                        status = f"TURN RIGHT (CW)"
                        status_color = (255, 0, 0)
                        print(f"[帧{frame_count:4d}] 顺时针转 - 角度: {angle:6.1f}° -> 速度: +{yaw_speed}")

                # 显示状态
                cv2.putText(frame, f"Status: {status}", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                # 角度条显示
                bar_x, bar_y = 10, 230
                bar_width, bar_height = 300, 20
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)

                # 阈值区域
                threshold_pixels = int((angle_threshold / 90) * (bar_width // 2))
                center_x = bar_x + bar_width // 2
                cv2.rectangle(frame, (center_x - threshold_pixels, bar_y),
                              (center_x + threshold_pixels, bar_y + bar_height), (0, 255, 0), -1)

                # 当前角度位置
                angle_pixels = int((angle / 90) * (bar_width // 2))
                angle_pos = center_x + angle_pixels
                cv2.line(frame, (angle_pos, bar_y), (angle_pos, bar_y + bar_height), (0, 255, 255), 3)

                # 角度标尺
                cv2.putText(frame, "-90", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, "0", (center_x - 5, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, "90", (bar_x + bar_width - 20, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1)

            else:
                # 没有检测到目标
                dog.stop()
                cv2.putText(frame, "NO TARGET DETECTED", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                print(f"[帧{frame_count:4d}] 未检测到目标")

            # 显示统计信息
            cv2.putText(frame, f"Frames: {frame_count}", (400, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {detection_count}", (400, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Corrections: {correction_count}", (400, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Aligned: {aligned_count}", (400, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 显示图像
            cv2.imshow('Yaw Correction Test', frame)
            cv2.imshow('Mask', vis_mask)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出测试")
                break
            elif key == ord('r'):
                print("重置统计数据")
                frame_count = 0
                detection_count = 0
                correction_count = 0
                aligned_count = 0
            elif key == ord(' '):
                # 切换颜色
                colors = list(color_dist.keys())
                current_idx = colors.index(target_color)
                target_color = colors[(current_idx + 1) % len(colors)]
                print(f"切换目标颜色为: {target_color}")
            elif key == ord('+') or key == ord('='):
                # 增加角度阈值
                angle_threshold = min(angle_threshold + 1, 30)
                print(f"角度阈值调整为: ±{angle_threshold}°")
            elif key == ord('-'):
                # 减小角度阈值
                angle_threshold = max(angle_threshold - 1, 1)
                print(f"角度阈值调整为: ±{angle_threshold}°")

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        # 清理资源
        dog.stop()
        cap.release()
        cv2.destroyAllWindows()

        # 打印测试统计
        print("\n=== 测试统计 ===")
        print(f"总帧数: {frame_count}")
        print(f"检测到目标: {detection_count} 次")
        print(f"执行修正: {correction_count} 次")
        print(f"对齐成功: {aligned_count} 次")
        if frame_count > 0:
            print(f"检测成功率: {detection_count / frame_count * 100:.1f}%")
        print("测试结束")


# 使用示例
if __name__ == "__main__":
    # 运行偏航修正测试
    # 参数说明：
    # target_color: 目标颜色 ('red', 'blue', 'green', 'yellow')
    # angle_threshold: 角度阈值（度）
    # yaw_speed: 偏航速度
    yaw_correction_test(target_color='red', angle_threshold=10, yaw_speed=8)
