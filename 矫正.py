import sys


sys.path.append('/home/pi/Desktop/main/package')

from xgolib import XGO
import time
from yolo_detector import initialize_model, predict_image


dog = XGO("/dev/ttyAMA0")

print("正在初始化教育库")
dog = XGO("xgolite")
version=dog.read_firmware()
if version[0]=='M':
    print('XGO-MINI')
    dog = XGO("xgomini")
    dog_type='M'
else:
    print('XGO-LITE')
    dog_type='L'
# 读取当前机器狗的偏航角作为相反方向的参考
for i in range(5):
    oppozite_yaw = dog.read_yaw()
left_yaw = oppozite_yaw + 90  # 左转90度的偏航角
print(left_yaw)
#初始化模型
initialize_model(model_path='weights/best.onnx')
print("初始化onnx成功")

#初始化摄像机

def adjust_x(vx, runtime):
    dog.move_x(vx)
    time.sleep(runtime)
    dog.move_x(0)


def adjust_y(vy, runtime):
    dog.move_y(vy)
    time.sleep(runtime)
    dog.move_y(0)

def adjust_to_yaw(target_yaw, turn_speed=10):
    """ 调整机器狗的姿态，使其恢复到初始姿态 """
    count = 0
    while True:
        current_yaw = dog.read_yaw()
        # print("当前角度：%d" %current_yaw)
        # print("目标的角度：%d" %target_yaw)
        # 计算姿态偏差
        yaw_diff = target_yaw - current_yaw
        # 处理 Yaw 角度的周期性
        if yaw_diff > 180:
            yaw_diff -= 360
        elif yaw_diff < -180:
            yaw_diff += 360
        # print("偏差：%d" %yaw_diff)
        if abs(yaw_diff) < 6:          #  一直是7
            break
        turn_speed = abs(yaw_diff) * 0.8333
        turn_speed = int(turn_speed)
        if turn_speed < 15:
            turn_speed = 15
        runtime = 1.0 * abs(yaw_diff) / turn_speed
        if runtime < 1:
            runtime = 1
        # print("speed:%d" %turn_speed)
        # print("time: %f" %runtime)
        # print("*"*30)
        # 根据偏差调整方向
        if yaw_diff > 0:
            # 向右转
            adjust_yaw(turn_speed, runtime)
        else:
            # 向左转
            adjust_yaw(-turn_speed, runtime)
        time.sleep(1)
        count += 1
        if count > 7:
            return

def cap_onnx()
    detected_label = capture_and_detect()
    if detected_label:
        print(f"检测到的最大物体: {detected_label}")
    else:
        print("未检测到任何物体")


if __name__ == '__main__':
    #先向前一点进行第一个木箱的识别
    adjust_x(15, 2)
    cap_onnx()
    time.sleep(0.5)
    adjust_y(-25, 3)  # 向右平移
    time.sleep(0.5)  # 短暂暂停
    cap_onnx()
    time.sleep(0.5)
    #返回
    adjust_y(50, 3)
    adjust_x(50, 2)
    time.sleep(0.5)
    cap_onnx()
    time.sleep(0.5)
    # adjust_to_yaw(oppozite_yaw, 10)  # 调整偏航角
    # adjust_x(25, 2)  # 向前移动
    # time.sleep(0.5)
    # adjust_to_yaw(oppozite_yaw, 10)  # 再次调整偏航角
    #
    # # -------------------------------------------------------------------------
    # adjust_x(-25, 1.5)  # 后退
    # time.sleep(0.5)
    #
    # # -----------------------------------------------------------
    # adjust_to_yaw(oppozite_yaw, 10)  # 调整偏航角
    # time.sleep(0.5)
    # adjust_y(20, 5)  # 向左平移
    # time.sleep(0.5)
    # adjust_to_yaw(oppozite_yaw)  # 调整偏航角
    # adjust_x(20, 5)  # 向前移动
    # time.sleep(0.5)
    # adjust_to_yaw(oppozite_yaw)  # 调整偏航角