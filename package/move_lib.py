from xgolib import XGO
import time
dog = XGO("/dev/ttyAMA0") 

def adjust_x(vx, runtime):
    """
    调整狗的x轴移动速度。该函数接受两个参数，一个是狗的x轴移动速度，另一个是运行时间。在给定的时间内，狗会按照指定的速度进行x轴移动，然后停止。
    
    Parameters:
    vx (int): 狗的x轴移动速度。
    runtime (float): 运行时间，单位为秒。
    
    Returns:
    无返回值。
    """
    dog.move_x(vx)
    time.sleep(runtime)
    dog.move_x(0)


def adjust_y(vy, runtime):
    """
    调整狗的垂直位置。该函数接受两个参数，一个是垂直速度（vy），另一个是运行时间（runtime）。首先，根据给定的垂直速度移动狗的位置，然后等待指定的运行时间，最后将狗的位置重置为0。
    
    Parameters:
    vy (int): 狗的垂直速度，可以是正数或负数。
    runtime (float): 狗在垂直方向上移动的时间，单位为秒。
    
    Returns:
    无返回值。
    """
    dog.move_y(vy)
    time.sleep(runtime)
    dog.move_y(0)


def adjust_yaw(vyaw, runtime):
    """
    调整狗的偏航角。该函数接受两个参数，一个是偏航角速度，另一个是运行时间。首先让狗以给定的偏航角速度转动，然后等待指定的运行时间，最后将狗的偏航角速度设为0。
    
    Parameters:
    vyaw (float): 狗的偏航角速度，单位为度/秒。
    runtime (float): 狗的偏航角速度维持的时间，单位为秒。
    
    Returns:
    无返回值。
    """
    dog.turn(vyaw)
    time.sleep(runtime)
    dog.turn(0)

def ready_for_grasp():
    # dog.reset()
    dog.attitude("p", 15)
    time.sleep(0.5)

def stand():
    dog.reset()
    time.sleep(0.5)

# 执行抓取动作
def grasp():
    dog.claw(0)
    time.sleep(0.5)
    dog.translation("x", 20)
    dog.motor(52, -55)
    time.sleep(0.5)
    dog.translation("z", 60)
    time.sleep(0.5)
    dog.motor(53, 80)
    dog.attitude("p", 20) 
    time.sleep(2)
    dog.claw(255)
    time.sleep(2)
    dog.reset()
    time.sleep(1)

    dog.motor(52, -55)
    time.sleep(1)


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



# 执行放置动作
def place():
    dog.translation("x", 20)
    dog.motor(52, -55)
    time.sleep(0.5)
    dog.translation("z",80)
    time.sleep(0.5)
    dog.motor(53, 80)
    dog.attitude("p", 8) 
    time.sleep(2)
    dog.claw(0)
    time.sleep(2)
    dog.reset()
    time.sleep(1)       
 
        
XGOorder = {
    "BATTERY": [0x01, 100],
    "PERFORM": [0x03, 0],
    "CALIBRATION": [0x04, 0],
    "UPGRADE": [0x05, 0],
    "SET_ORIGIN": [0x06, 1],
    "FIRMWARE_VERSION": [0x07],
    "GAIT_TYPE": [0x09, 0x00],
    "BT_NAME": [0x13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "UNLOAD_MOTOR": [0x20, 0],
    "LOAD_MOTOR": [0x20, 0],
    "VX": [0x30, 128],
    "VY": [0x31, 128],
    "VYAW": [0x32, 128],
    "TRANSLATION": [0x33, 0, 0, 0],
    "ATTITUDE": [0x36, 0, 0, 0],
    "PERIODIC_ROT": [0x39, 0, 0, 0],
    "MarkTime": [0x3C, 0],
    "MOVE_MODE": [0x3D, 0],
    "ACTION": [0x3E, 0],
    "MOVE_TO": [0x3F, 0, 0],
    "PERIODIC_TRAN": [0x80, 0, 0, 0],
    "MOTOR_ANGLE": [0x50, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
    "MOTOR_SPEED": [0x5C, 1],
    "MOVE_TO_MID": [0x5F, 1],
    "LEG_POS": [0x40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IMU": [0x61, 0],
    "ROLL": [0x62, 0],
    "PITCH": [0x63, 0],
    "TEACH_RECORD": [0x21, 0],
    "TEACH_PLAY": [0x22, 0],
    "TEACH_ARM_RECORD": [0x23, 0],
    "TEACH_ARM_PLAY": [0x24, 0],
    "YAW": [0x64, 0],
    "CLAW": [0x71, 0],
    "ARM_MODE": [0x72, 0],
    "ARM_X": [0x73, 0],
    "ARM_Z": [0x74, 0],
    "ARM_SPEED": [0x75, 0],
    "ARM_THETA": [0x76, 0],
    "ARM_R": [0x77, 0],
    "OUTPUT_ANALOG": [0x90, 0],
    "OUTPUT_DIGITAL": [0x91, 0],
    "LED_COLOR": [0x69, 0, 0, 0]
}

def send(key, index=1, len=1):
    mode = 0x01
    order = XGOorder[key][0] + index - 1
    value = []
    value_sum = 0
    for i in range(0, len):
        value.append(XGOorder[key][index + i])
        value_sum = value_sum + XGOorder[key][index + i]
    sum_data = ((len + 0x08) + mode + order + value_sum) % 256
    sum_data = 255 - sum_data
    tx = [0x55, 0x00, (len + 0x08), mode, order]
    tx.extend(value)
    tx.extend([sum_data, 0x00, 0xAA])
    dog.ser.write(tx)
    if dog.verbose:
        print("tx_data: ", tx)

def rider_led(index, color):
    XGOorder["LED_COLOR"][0] = 0x68 + index
    XGOorder["LED_COLOR"][1:4] = color
    send("LED_COLOR", len=3)

def led(color):
    if color == "red":
        color = [255, 0, 0]
    elif color == "green":
        color = [0, 255, 0]
    elif color == "blue":
        color = [0, 0, 255]
    else:
        color = [0, 0, 0]
    for i in range(1, 4):
        rider_led(i, color)

# 狗1执行放置动作
def place_one():
    dog.translation("x", 20)
    dog.motor(52, -55)
    time.sleep(0.5)
    dog.translation("z",80)
    time.sleep(0.5)
    dog.motor(53, 80)
    dog.attitude("p", 15) 
    time.sleep(2)
    dog.claw(0)
    time.sleep(2)
    dog.reset()
    time.sleep(1)

# 狗2执行放置动作
def place_two():
    dog.translation("x", 20)
    dog.motor(52, -55)
    time.sleep(0.5)
    dog.translation("z",80)
    time.sleep(0.5)
    dog.motor(53, 80)
    dog.attitude("p", 10) 
    time.sleep(2)
    dog.claw(0)
    time.sleep(2)
    dog.reset()
    time.sleep(1)

if __name__ == '__main__':
    led('red')
    input()
    led('blue')
    input()
    led('green')
    input()
    led('off')