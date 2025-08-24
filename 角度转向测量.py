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
        current_distance,angle = run_distance(300,1200)
        if current_distance is None:
            continue
        error = aim_distance-current_distance
        print(f"当前距离: {current_distance:.1f}")
        if abs(error)<=15:
            break
        if error<0:
            runtime = calculate_dog_runtime(error/10, -speed)
            print(runtime)
            print(f"当前error: {error:.1f}")
            adjust_x(-speed,runtime)
        else:
            runtime = calculate_dog_runtime(error/10, speed)
            print(runtime)
            print(f"当前error: {error:.1f}")
            adjust_x(speed,runtime)
        if num>=10:
            return False
        time.sleep(6)
        num=num+1
    print("SSSSSS")
    return True


def calculate_angle_difference_two(large_angle, small_angle):
    """
    通过先转大角度再转小角度的方式实现精确角度调整

    参数:
    large_angle: 先转的大度数
    small_angle: 再转的小度数（用于精确调整）

    返回:
    angle_diff: 车偏离量的差值
    """
    # 执行大角度转向、
    _, angle_after_large = run_distance(300, 1200)
    if angle_after_large is None:
        return None

    controller.turn_by_angle(large_angle)
    time.sleep(1)  # 等待转向稳定


    controller.turn_by_angle(small_angle)
    time.sleep(1)  # 等待转向稳定

    # 获取最终的偏离值
    _, angle_after_small = run_distance(300, 1200)
    if angle_after_small is None:
        return None

    # 计算偏离量的差值
    angle_diff = angle_after_large - angle_after_small

    print(f"偏离量差值: {angle_diff}")

    return angle_diff
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
        print(f"角度误差：{angle_error}")
        if abs(angle_error)<=3:
            print(all_in)
            print("《《《《《5")
            break
        controller.turn_by_angle(angle_error+2.5)
        time.sleep(1)  # 等待转向稳定
        # controller.turn_by_angle(small_angle)
        # time.sleep(1)  # 等待转向稳定
        # # 获取最终的偏离值
        print(f"角度差值: {angle_error}")
        if num>=10:
            print("矫正次数大于10失败")
            return False
    print("矫正成功")
    return True


def calculate_angle_difference_one(angle):
    """
    通过先转大角度再转小角度的方式实现精确角度调整

    参数:
    large_angle: 先转的大度数
    small_angle: 再转的小度数（用于精确调整）

    返回:
    angle_diff: 车偏离量的差值
    """
    # 执行大角度转向
    _, angle_before_large = run_distance(300, 1200)
    if angle_before_large is None:
        return None
    controller.turn_by_angle(angle)
    time.sleep(1)  # 等待转向稳定
    # 获取转向后的偏离值
    _, angle_after_large = run_distance(300, 1200)
    if angle_after_large is None:
        return None

    # 计算偏离量的差值
    angle_diff = angle_after_large - angle_before_large

    print(f"偏离量差值: {angle_diff}")

    return angle_diff
if __name__ == '__main__':
    # controller.turn_by_angle(90)
    # correct_turn()
    # location(300)
    # controller.turn_by_angle(90)
    # location(500)
    # while True:
    #     L_angle=input("LLL")
    #     S_angle=input("SSS")
    #     diff=calculate_angle_difference_two(L_angle,S_angle)
    #     print(f"转动：{diff}")
    # while False:
    #     angle=input("AAA")
    #     diff=calculate_angle_difference_one(angle)
    #     print(f"转动：{diff}")
    angle_correct()
    #controller.turn_by_angle(90)
    #correct_turn()