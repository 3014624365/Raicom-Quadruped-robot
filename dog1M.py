import sys
sys.path.append('/home/pi/Desktop/code/package')
import time
import cv2
import os
import numpy as np
from xgolib import XGO
from move_lib import *
from correct_turn_dog import *
from math_time import calculate_dog_runtime
from raber import *
from grob_and_place.py import *

#初始化准备
dog = XGO("/dev/ttyAMA0")
print("正在初始化教育库")
dog = XGO("xgolite")
# ==========================================================
controller = AdaptiveYawController(dog)
start_yaw = controller.get_stable_yaw(readings=7)  # 增加读数次数
start_yaw_norm = start_yaw % 360
controller.set_reference_yaw(start_yaw)
# ===========================================================
#初始化模型
initialize_model(model_path='weights/best.onnx')
print("初始化onnx成功")
#============================================================
# 打印电池电量
print(f"电池电量: {dog.read_battery()}%")
def fangzhi():
    # 设置机身俯仰角、偏航角、滚转角
    dog.attitude('p', 40)
    # 设置机械臂位置
    dog.arm(140, 0)  # 机械臂末端位于基座正上方100mm处
    dog.arm(90, 90)  # 机械臂大臂垂直于身体，小臂水平于身体
    dog.arm(100, 0)  # 机械臂末端在摄像头正前方
    dog.arm(100, -60)  # 机械臂末端下探抓取
    time.sleep(2)  # 等待2秒
    # 放下物体
    dog.claw(10)
    time.sleep(1)  # 等待1秒
    # 收回机械臂，机械臂末端回到摄像头正前方
    dog.arm(100, 0)
    # 恢复站立
    dog.attitude('p', 0)
    dog.reset()
    dog.claw(100)
    time.sleep(1)


def cap_onnx():
    detected_label = capture_and_detect()
    if detected_label:
        print(f"检测到的最大物体: {detected_label}")
    else:
        print("未检测到任何物体")


#距离矫正
def location(distance_error,aim_distance, speed=10):
    num = 0
    while True:
        current_distance, angle = run_distance(300, 1200)
        if current_distance is None:
            continue
        error = aim_distance - current_distance
        if abs(error) <= distance_error:
            break
        if error < 0:
            runtime = calculate_dog_runtime(error / 10, -speed)
            adjust_x(-speed, runtime)
        else:
            runtime = calculate_dog_runtime(error / 10, speed)
            adjust_x(speed, runtime)
        if num >= 10:
            return False
        time.sleep(6)
        num = num + 1
    print("矫正成功")
    return True

#后方角度矫正
def angle_correct(error_th,min_angle=150,max_angle=210):
    num=0
    while True:
        all_in=0
        i=0
        while i<2:
            _, angle_error = run_distance(300, 1200,min_angle,max_angle)
            if angle_error is None:
                continue
            else:
                all_in=all_in+angle_error
            i=i+1
        angle_error=all_in/2
        if abs(angle_error)<=error_th:
            break
        if angle_error < 0:
            controller.turn_by_angle(angle_error + 2.5)
        else:
            controller.turn_by_angle(angle_error)
        if num>=10:
            return False
    print("矫正成功")
    return True

def angle_correct_45(error_th,aim_angle=45):
    controller.turn_by_angle(aim_angle)
    num=0
    while True:
        _, angle_error = run_distance(300, 1200,,210-aim_angle)
        if angle_error is None:
            continue
        angle_error=all_in/2-aim_angle
        if abs(angle_error)<=error_th:
            break
        if angle_error < 0:
            controller.turn_by_angle(angle_error + 2.5)
        else:
            controller.turn_by_angle(angle_error)
        if num>=10:
            return False
    print("矫正成功")
    return True

def wall_angle_correct(min_angle, max_angle):
    num = 0
    while True:
        all_in = 0
        i = 0
        while i < 2:
            _, angle = run_distance(300, 1200, min_angle, max_angle)
            print(angle)
            if angle is None:
                continue
            elif angle < 0:
                angle = angle + 90
            else:
                angle = angle - 90
            all_in = all_in + angle
            i = i + 1
        angle_error = (all_in / 2)
        if abs(angle_error) <= 3:
            break
        if angle_error < 0:
            controller.turn_by_angle(angle_error + 2.5)
        else:
            controller.turn_by_angle(angle_error)
        if num >= 10:
            return False
    print("矫正成功")
    return True
if __name__ == '__main__':
        # # 第一个测试识别===========================
        # print("\n---- 第一个检测点 ----")
        # adjust_x(5, 2)
        # time.sleep(0.3)
        # cap_onnx()
        # #第二测试识别=============================
        # controller.turn_by_angle(-86)
        # controller.turn_by_angle(90)
        # time.sleep(0.3)
        # cap_onnx()
        # #第三测试识别============================
        # print("执行左转90度")
        # controller.turn_by_angle(93)
        # time.sleep(0.3)
        # adjust_x(20, 5.25)
        # time.sleep(0.3)
        # print("执行右90度")
        # controller.turn_by_angle(-90)
        # time.sleep(0.3)
        # adjust_x(20, 4)
        # time.sleep(0.3)
        # cap_onnx()
        # print("执行转90度")
        # controller.turn_by_angle(-90)
        # time.sleep(0.3)
        # #前往抓取区================================
        # print("完成所有检测，开始抓取")
        # adjust_x(22, 3)
        # time.sleep(0.3)
        # print("执行左转90度")
        # controller.turn_by_angle(92)
        # time.sleep(0.3)
        # # 前进
        # adjust_x(22, 3)
        # time.sleep(0.3)
        # print("执行转90度")
        # controller.turn_by_angle(-90)
        # time.sleep(0.3)
        # #前进
        # adjust_x(11, 6)
        # time.sleep(0.5)
        # print("执行转90度")
        # controller.turn_by_angle(90)
        # time.sleep(0.5)
        # adjust_x(6,4)
        # time.sleep(0.3)
        # #使用激光雷达矫正
        # angle_correct(3,0)
        # time.sleep(0.3)
        # pick_cuboid()
        # # 返回================================================
        # adjust_x(-7, 3)
        # time.sleep(0.3)
        # print("执行右转90度")
        # controller.turn_by_angle(-90)
        # time.sleep(0.5)
        # # 使用激光雷达矫正
        # angle_correct(5)
        # # 使用激光雷达定位
        # location(400)
        # angle_correct_45(3,45)
        # adjust_x(-7, 3)
        # #===============================================
        wall_angle_correct(230,310)
        location(30,300)
        controller.turn_by_angle(135)
        angle_correct(3)
        location(50,500)
