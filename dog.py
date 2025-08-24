import sys
import time
import cv2
import os
import numpy as np
from xgolib import XGO
from move_lib import *
from correct_turn_dog import *
from raber_located import *


# from new_yolo import *
def adjust_x(vx, runtime):
    dog.move_x(vx)
    time.sleep(runtime)
    dog.move_x(0)


def adjust_y(vy, runtime):
    dog.move_y(vy)
    time.sleep(runtime)
    dog.move_y(0)


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


def get_picture():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for i in range(10):
        rec, img = cap.read()
    cap.release()
    return img

def cap_onnx():
    img = get_picture()
    cv2.imwrite("raw1.jpg", img)
    # 执行识别和播报语音
    result = detect_and_speak(img, yolo, store=1)
    print(f"识别结果类别: {result}")


if __name__ == '__main__':
    try:
        # 初始化机器狗
        print("初始化机器狗...")
        dog = XGO("xgolite")  # 或者 XGO("xgolite")，取决于您的设备
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

        # # 初始化模型
        # weights_path = '/home/pi/dog/best.pt'
        # yolo = load_model(weights_path)

        if 1:
            print("初始化onnx成功")

            # 打印电池电量
            print(f"电池电量: {dog.read_battery()}%")

            # # 第一个移动和识别序列
            # print("\n---- 第一个检测点 ----")
            # adjust_x(5, 2)
            # # # 第一个识别点
            # # cap_onnx()
            # # time.sleep(0.5)

            # # 右转90度
            # print("执行右转90度")
            # controller.turn_by_angle(-86)  # 使用控制器而不是直接dog.turn
            # time.sleep(0.5)

            # # 继续移动 - 使用增强的自适应转向控制
            # # 前进
            # adjust_x(22, 3)
            # time.sleep(0.5)

            # # 左转90度
            # print("执行转左90度")
            # controller.turn_by_angle(90)  # 使用控制器而不是直接dog.turn
            # time.sleep(0.5)

            # # 前进
            # adjust_x(24, 2)

            # # # 第二个识别点
            # # print("\n---- 第二个检测点 ----")
            # # cap_onnx()
            # # time.sleep(0.5)

            # # 返回路径移动 - 使用增强的自适应转向控制
            # # 后退
            # adjust_x(-5, 3)
            # time.sleep(0.5)

            # # 左转45度
            # print("执行左转90度")
            # controller.turn_by_angle(93)  # 使用控制器而不是直接dog.turn
            # time.sleep(0.5)

            # # 前进
            # adjust_x(20, 5.25)
            # time.sleep(0.5)

            # # 右转90度
            # print("执行右90度")
            # controller.turn_by_angle(-90)  # 使用控制器而不是直接dog.turn
            # time.sleep(0.5)

            # # 前进
            # adjust_x(20, 4)
            # time.sleep(0.5)

            # # # 第三个识别点
            # # print("\n---- 第三个检测点 ----")
            # # cap_onnx()
            # # time.sleep(0.5)

            # # # 完成最后的移动 - 使用增强的自适应转向控制
            # # # 右转90度
            # print("执行转90度")
            # controller.turn_by_angle(-90)  # 使用控制器而不是直接dog.turn
            # time.sleep(0.5)

            # # # 前进
            # adjust_x(22, 3)
            # time.sleep(0.5)

            # # # 左转90度
            # print("执行左转90度")
            # controller.turn_by_angle(92)  # 使用控制器而不是直接dog.turn
            # time.sleep(0.5)

            # # # 前进
            # adjust_x(22, 3)
            # time.sleep(0.5)

            # print("执行转90度")
            # controller.turn_by_angle(-90)  # 使用控制器而不是直接dog.turn
            # time.sleep(0.5)

            # # # 前进
            # adjust_x(11, 6)
            # time.sleep(0.5)

            # print("执行转90度")
            # controller.turn_by_angle(90)  # 使用控制器而不是直接dog.turn
            # time.sleep(0.5)

            # adjust_x(6,4)
            # time.sleep(0.5)

            # #使用激光雷达矫正
            # angle_correct()

            # #执行抓取
            # dog_color_grab(target_color="red")
            # time.sleep(0.5)
            os.system("python3 grab_and_place.py")

            # # 后退
            adjust_x(-7, 3)
            time.sleep(0.5)

            # 右转90度
            print("执行右转90度")
            controller.turn_by_angle(-90)  # 使用控制器而不是直接dog.turn
            time.sleep(0.5)

            # 使用激光雷达矫正
            angle_correct()
            # 使用激光雷达定位
            location(400)
            # 左转90度
            print("执行左转90度")
            controller.turn_by_angle(90)  # 使用控制器而不是直接dog.turn
            time.sleep(0.5)

            # 使用激光雷达定位
            location(400)
            # 使用激光雷达矫正
            # angle_correct()
            # # 左转90度
            # print("执行左转90度")
            controller.turn_by_angle(100)  # 使用控制器而不是直接dog.turn
            time.sleep(0.5)

            # # 前进
            adjust_x(22, 3)
            time.sleep(0.5)

            # # 左转90度
            print("执行左转90度")
            controller.turn_by_angle(95)  # 使用控制器而不是直接dog.turn
            time.sleep(0.5)

            # # 前进
            adjust_x(24, 4)
            time.sleep(0.5)

            # # 右转90度
            print("执行右转90度")
            controller.turn_by_angle(-75)  # 使用控制器而不是直接dog.turn
            time.sleep(0.5)

            # # 使用激光雷达矫正
            # angle_correct()

            # # 前进
            adjust_x(20, 4)
            time.sleep(0.5)

            fangzhi()

            adjust_x(-10, 3)
            time.sleep(0.5)
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
