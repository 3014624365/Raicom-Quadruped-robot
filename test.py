import sys

sys.path.append('/home/pi/Desktop/code/package')
from correct_turn_dog import *
from raber import *
from math_time import calculate_dog_runtime
from xgolib import XGO
import time

dog = XGO("/dev/ttyAMA0")
print("正在初始化教育库")
dog = XGO("xgolite")
version = dog.read_firmware()
if version[0] == 'M':
    print('XGO-MINI')
    dog = XGO("xgomini")
    dog_type = 'M'
else:
    print('XGO-LITE')
    dog_type = 'L'
# 读取当前机器狗的偏航角作为相反方向的参考
for i in range(5):
    oppozite_yaw = dog.read_yaw()
# 初始化矫正转向的初始化
# ==========================================================
controller = AdaptiveYawController(dog)
start_yaw = controller.get_stable_yaw(readings=7)  # 增加读数次数
start_yaw_norm = start_yaw % 360
controller.set_reference_yaw(start_yaw)
# ===========================================================
# 初始化摄像机

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


def location(aim_distance,speed=10):
    print("开始矫正")
    num=0
    while True:
        current_distance = run_distance()
        error = current_distance - aim_distance
        print(f"当前距离: {current_distance:.1f}°")
        if error<=15:
            break
        runtime = calculate_dog_runtime(error, speed)
        adjust_x(speed,runtime)
        if num>=10:
            return False
        return True


if __name__ == '__main__':
    # controller.turn_by_angle(90)
    # correct_turn()
    location(300)
    controller.turn_by_angle(90)
    location(500)