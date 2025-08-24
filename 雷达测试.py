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


#距离矫正
def location(aim_distance, speed=10):
    num = 0
    while True:
        current_distance, angle = run_distance(300, 1200)
        if current_distance is None:
            continue
        error = aim_distance - current_distance
        if abs(error) <= 15:
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

#平板矫正
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
#后方角度矫正
def angle_correct():
    num=0
    while True:
        all_in=0
        i=0
        while i<2:
            _, angle_error = run_distance(300, 1200)
            if angle_error is None:
                continue
            else:
                all_in=all_in+angle_error
            i=i+1
        angle_error=all_in/2
        if abs(angle_error)<=3:
            break
        if angle_error < 0:
            controller.turn_by_angle(angle_error + 2.5)
        else:
            controller.turn_by_angle(angle_error)
        if num>=10:
            return False
    print("矫正成功")
    return True



if __name__ == '__main__':
    # controller.turn_by_angle(90)
    # correct_turn()
    # location(300)
    # controller.turn_by_angle(90)
    # location(500)
    #230，310是向前的角度矫正
    wall_angle_correct(230, 310)
    #50，230是回来的角度矫正
    wall_angle_correct(50, 230)