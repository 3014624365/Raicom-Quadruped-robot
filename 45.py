import sys
import time
import cv2
import os
import numpy as np
from xgolib import XGO
from move_lib import *
from correct_turn_dog import *


def adjust_x(vx, runtime):
    dog.move_x(vx)
    time.sleep(runtime)
    dog.move_x(0)


def adjust_y(vy, runtime):
    dog.move_y(vy)
    time.sleep(runtime)
    dog.move_y(0)


def correct_turn():
    current_yaw = controller.get_stable_yaw()
    current_yaw_norm = current_yaw % 360
    # 检查与初始角度的差异
    angle_diff = controller.calculate_angle_diff(current_yaw_norm, start_yaw_norm)
    print(f"当前角度与初始角度的差异: {angle_diff:.1f}°")
    controller.check_and_correct_orientation(start_yaw, error_threshold=5)


# 用于拍摄和检测的函数
# def cap_onnx(show_image=False):
#     """拍摄并显示识别结果"""
#     detected_label = capture_and_detect(show_image=show_image)
#     if detected_label:
#         print(f"检测到的最大物体: {detected_label}")
#     else:
#         print("未检测到任何物体")
#     return detected_label


if __name__ == '__main__':
    try:
        # 初始化机器狗
        print("初始化机器狗...")
        dog = XGO("/dev/ttyAMA0")  # 或者 XGO("xgolite")，取决于您的设备
        print("机器狗初始化成功")

        # 初始化自适应陀螺仪控制器
        controller = AdaptiveYawController(dog)

        # 确保机器狗稳定
        print("等待机器狗完全稳定...")
        time.sleep(1.5)

        # 读取当前机器狗的偏航角作为起始方向的参考
        start_yaw = controller.get_stable_yaw(readings=7)  # 增加读数次数
        start_yaw_norm = start_yaw % 360
        print(f"初始偏航角: {start_yaw} (标准化={start_yaw_norm:.1f}°)")

        # 保存这个初始方向作为参考
        controller.set_reference_yaw(start_yaw)

        # 初始化模型
        print("初始化模型中...")
        if 1:
            print("初始化onnx成功")

            # 打印电池电量
            print(f"电池电量: {dog.read_battery()}%")

            # print("执行转90度")
            # controler.turn_by_angle(89)  # 使用控制器而不是直接dog.turn
            # time.sleep(0.5)

            # # # 前进
            # adjust_x(11, 6)
            # time.sleep(0.5)

            # # # 转90度
            # print("执行转90度")
            # controller.turn_by_angle(89)  # 使用控制器而不是直接dog.turn
            # time.sleep(0.5)

            # # # 前进
            # adjust_x(11, 8)
            # time.sleep(0.5)

            # # # 转90度
            # print("执行转90度")
            # controller.turn_by_angle(-91)  # 使用控制器而不是直接dog.turn
            # time.sleep(0.5)

            # # # 前进
            # adjust_x(11, 6)
            # time.sleep(0.5)

        else:
            print("模型初始化失败")

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序出错: {str(e)}")
        import traceback

        traceback.print_exc()  # 打印详细错误堆栈
    finally:
        # 释放资源
        if 'dog' in locals() and dog is not None:
            dog.stop()
        print("程序结束")
