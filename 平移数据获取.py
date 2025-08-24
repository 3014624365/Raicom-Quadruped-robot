import sys

sys.path.append('/home/pi/Desktop/code/package')
from correct_turn_dog import *
from raber import *
from math_time import calculate_dog_runtime
from xgolib import XGO
import time
import csv

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


def collect_data():
    data = []

    print("请依次输入数据，输入非数字或按Ctrl+C停止")

    while True:
        try:
            speed = float(input("输入速度: "))
            runtime = float(input("输入时间: "))
            # 调用你的函数
            adjust_y(speed, runtime)
            shiji_distance = float(input("输入实际距离: "))

            # 保存数据
            data.append([speed, runtime, shiji_distance])
            print(f"已记录: 速度={speed}, 时间={runtime}, 实际距离={shiji_distance}")
            print("-" * 30)

        except ValueError:
            print("输入无效，结束数据收集")
            break
        except KeyboardInterrupt:
            print("\n\n手动停止，结束数据收集")
            break

    # 保存到CSV文件
    if data:
        with open('data.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['速度', '时间', '实际距离'])  # 表头
            writer.writerows(data)
        print(f"数据已保存到 data.csv，共 {len(data)} 条记录")
    else:
        print("没有数据需要保存")


if __name__ == '__main__':
    collect_data()