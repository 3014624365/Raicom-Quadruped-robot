# test_gyro.py

import time
import sys

sys.path.append('/home/pi/Desktop/main/package')  # 修改为你的实际路径

from xgolib import XGO
import gyro_controller  # 导入我们封装的模块

# 初始化机器狗
print("初始化机器狗...")
dog = XGO("/dev/ttyAMA0")  # 或者根据您的设备修改
# 打印电池电量
print(f"电池电量: {dog.read_battery()}%")

print("机器狗初始化成功")
# 初始化陀螺仪控制器
controller = gyro_controller.initialize_controller(dog)


def corurt_turn(angle):
    try:
        time.sleep(1.5)
        success, actual = gyro_controller.turn(angle)
        print(f"左转完成, 成功: {success}, 实际转角: {actual:.1f}°")
        time.sleep(1)

        # 检查并校正当前角度
        print("检查并校正当前角度")
        corrected = gyro_controller.check_and_correct()
        print(f"校正结果: {'成功' if corrected else '失败'}")



if __name__ == "__main__":
    corurt_turn()

