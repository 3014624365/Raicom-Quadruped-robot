import math
import time
import cv2
import numpy as np
import xgolib
from move_lib import *


lower_yellow = np.array([17, 231, 227])  # 黄色范围的下界
upper_yellow = np.array([23, 255, 255])  # 黄色范围的上界

# lower_red = np.array([0, 50, 50])
# upper_red = np.array([10, 255, 255])

red_lower1 = [0, 50, 50]  # 第一个红色范围的下限
red_upper1 = [10, 255, 255]  # 第一个红色范围的上限
red_lower2 = [170, 50, 50]  # 第二个红色范围的下限
red_upper2 = [180, 255, 255]  # 第二个红色范围的上限
color_ranges_red = [[red_lower1, red_upper1], [red_lower2, red_upper2]]

# lower_green = np.array([50, 120, 0])
# upper_green = np.array([80, 255, 255])

# lower_blue = np.array([100, 150, 50])
# upper_blue = np.array([140, 255, 255])

# 给定的视频帧（frame）中过滤出特定颜色范围内的图像部分
def filter_img(frame, color):

    # b, g, r = cv2.split(frame)
    # frame = cv2.merge((r, g, b))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_lower = np.array(color[0])
    color_upper = np.array(color[1])
    mask = cv2.inRange(hsv, color_lower, color_upper)
    img_mask = cv2.bitwise_and(frame, frame, mask=mask)
    return img_mask

# 对于红色
def filter_img_red(frame, color_ranges):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = [cv2.inRange(hsv, np.array(color_range[0]), np.array(color_range[1])) for color_range in color_ranges]
    combined_mask = cv2.bitwise_or(*masks)
    img_mask = cv2.bitwise_and(frame, frame, mask=combined_mask)
    return img_mask


# 在图像中检测轮廓（contours），并返回这些轮廓以及用于检测轮廓的边缘图像
def detect_contours(frame):

    # CANNY_THRESH_1 = 16
    # CANNY_THRESH_2 = 120
    # edges = cv2.Canny(frame, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=3)
    edges = cv2.erode(edges, kernel, iterations=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, edges

def detect_color():

    dog.attitude("p", 15)
    time.sleep(0.5)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FPS, 30)
        # 定义颜色范围

    lower_red_1 = np.array([0, 50, 50])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 50, 50])
    upper_red_2 = np.array([180, 255, 255])

    lower_green = np.array([65, 120, 0])
    upper_green = np.array([80, 255, 255])

    lower_blue = np.array([90, 50, 0])
    upper_blue = np.array([110, 255, 255])
    while 1:
        # get a frame
        ret, frame = cap.read()

        ### 只识别下面70%区域
        # 获取图像尺寸
        height, width, _ = frame.shape
        
        # 计算下部70%区域的起始行位置
        start_row = int(height * 0.3)
        
        # 计算左右各去掉10%后的新宽度
        start_col = int(width * 0.2)  # 左侧10%
        end_col = int(width * 0.8)    # 右侧10%

        # 裁剪图像到下部70%，并去掉左右各10%
        cropped_image = frame[start_row:height, start_col:end_col]

        # 将BGR图像转换为HSV
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)


        # 创建颜色掩膜
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # 对掩膜进行形态学操作，如开运算和闭运算，以去除噪声
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        # 计算每个颜色掩膜中非零像素的数量
        blue_count = cv2.countNonZero(blue_mask)
        red_count = cv2.countNonZero(red_mask)
        green_count = cv2.countNonZero(green_mask)

        # 确定哪种颜色占据更多的像素
        max_count = max(blue_count, red_count, green_count)
        if max_count == blue_count:
            color = 'blue'
            break
        elif max_count == red_count:
            color = 'red'
            break
        else:
            color = 'green'
            break
        
    cap.release()
    cv2.destroyAllWindows()
    dog.reset()
    time.sleep(0.5)
    return color

# 从给定的图像帧（frame）中检测并处理最大的轮廓（假设这个轮廓代表了一个“块”或“矩形”对象），并返回与该轮廓相关的几个参数以及更新后的图像帧
def detect_block(contours, frame):
    flag = False
    length, width, angle, s_x, s_y = 0, 0, 0, 0, 0
    for i in range(0, len(contours)):
        if cv2.contourArea(contours[i]) < 5000:  #改动
            continue
        rect = cv2.minAreaRect(contours[i])
        
        # if 0.44 < rect[1][0] / rect[1][1] < 2.5: 
        #     continue

        # if 0.5 < rect[1][0] / rect[1][1] < 2: 
        #     continue

        
        if not flag:
            if rect[2] > 45:
                length = rect[1][0]
                width = rect[1][1]
                angle = rect[2]
                
            else:
                length = rect[1][1]
                width = rect[1][0]
                angle = rect[2]
            s_x = rect[0][1]  # s_代表屏幕坐标系
            s_y = rect[0][0]
            flag = True
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # 绘制最小外接矩形
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 5)

            break

        # else:  # 识别出两个及以上的矩形退出
        #     flag = False
        #     break
    return flag, length, width, angle, s_x, s_y, frame







# 根据当前机器人的位置 (m_x, m_y) 和期望的目标位置 (des_x, des_y) 来调整机器人的位置
# def adjust(m_angle, m_x, m_y, des_x=1050, des_y=960):

def adjust(m_angle, m_x, m_y, opposite_yaw, des_x = 1200, des_y = 980):
    origin_yaw = opposite_yaw
    # dog.gait_type("slow_trot")

    err_x = des_x - m_x
    err_y = des_y - m_y
    print(err_x, err_y)

    if abs(err_y) > 500:
        adjust_y(math.copysign(12, err_y), max(0.3, min(abs(err_y) / 200, 1.5)))
        print("左右大移动")
    else:
        if abs(err_x) < 70:
            if abs(err_y) < 40:
                grasp()
                return True
            else:
                dog.gait_type("slow_trot")
                # adjust_y(math.copysign(12, err_y), max(0.3, min(abs(err_y) / 200, 0.5)))

                adjust_y(math.copysign(12, err_y), max(0.3, min(abs(err_y) / 200, 1.5)))
                print("左右移动")
                

        else:
            dog.gait_type("trot")
            adjust_x(math.copysign(10, err_x), max(0.7, min(abs(err_x) / 150, 1.5)))
            print("前后移动")
    
    ################# 纠偏
    dog.gait_type("trot")
    adjust_to_yaw(origin_yaw, 10)

    time.sleep(0.3)
    return False


def adjust_place(m_angle, m_x, m_y, opposite_yaw, des_x = 1200, des_y = 980):
    origin_yaw = opposite_yaw
    # dog.gait_type("slow_trot")

    err_x = des_x - m_x
    err_y = des_y - m_y
    #print(err_x, err_y)

    if abs(err_y) > 500:
        adjust_y(math.copysign(12, err_y), max(0.3, min(abs(err_y) / 200, 1.5)))
    else:
        if abs(err_x) < 70:
            if abs(err_y) < 40:
                place_two()
                return True
            else:
                dog.gait_type("slow_trot")
                
                adjust_y(math.copysign(12, err_y), max(0.3, min(abs(err_y) / 200, 1.5)))
                #print("左右移动")
                

        else:
            dog.gait_type("trot")

            adjust_x(math.copysign(10, err_x), max(0.7, min(abs(err_x) / 150, 1.5)))

            #print("前后移动")
    
    ################# 纠偏
    dog.gait_type("trot")
    adjust_to_yaw(origin_yaw, 10)

    time.sleep(0.3)
    return False

def seek_pick(opposite_yaw, color='blue'):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    if color == 'blue':
        min_blue = [90, 50, 0]
        max_blue = [110, 255, 255]
    
    elif color == 'green':
        min_blue = [65, 150, 0]
        max_blue = [80, 255, 255]

    m_angle, m_x, m_y = 0, 0, 0
    count = 0
    COUNT_MAX = 20      # 原来为10   15
    ready_for_grasp()
    j = 0
    while 1:
        # get a frame
        ret, frame = cap.read()
        
        if not ret:  
            print("Error reading frame")  
        else:  
            frame = cv2.resize(frame, (1920, 1440))

        if color == 'red':
            frame_filter = filter_img_red(frame, color_ranges_red)
        else:
            frame_filter = filter_img(frame, [min_blue, max_blue])
        # if j % 5 == 0:
        #     print("saved>>>>>")
        #     string = 'pick_log/' + str(j) + '.jpg'
        #     cv2.imwrite(string, frame)

        counters, frame = detect_contours(frame_filter)

        flag, length, width, angle, s_x, s_y, frame = detect_block(counters, frame_filter)
        # if j % 5 == 0:
        #     print("saved>>>>>")
        #     string = 'pick_log/' + str(j) + '.jpg'
        #     cv2.imwrite(string, frame)
        j+=1
        if flag:
            count += 1
            m_angle = (count - 1) / count * m_angle + angle / count
            m_x = (count - 1) / count * m_x + s_x / count
            m_y = (count - 1) / count * m_y + s_y / count
            

        if count == COUNT_MAX:
            # print("我的位置",m_x,m_y)
            
            count = 0
            res = adjust(m_angle, m_x, m_y, opposite_yaw=opposite_yaw)
            if res:

                break

    cap.release()
    cv2.destroyAllWindows()


def seek_place(opposite_yaw, color='blue'):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if color == 'blue':
        min_blue = [90, 50, 0]
        max_blue = [110, 255, 255]
    elif color == 'green':
        min_blue = [65, 150, 0]
        max_blue = [80, 255, 255]

    m_angle, m_x, m_y = 0, 0, 0
    count = 0
    COUNT_MAX = 20      # 原来为10
    ready_for_grasp()
    j = 0
    while 1:
        # get a frame
        ret, frame = cap.read()
        
        if not ret:  
            print("Error reading frame")  
        else:  
            frame = cv2.resize(frame, (1920, 1440))

        if color == 'red':
            frame_filter = filter_img_red(frame, color_ranges_red)
        else:
            frame_filter = filter_img(frame, [min_blue, max_blue])
        if j % 5 == 0:
            print("saved>>>>>")
            string = 'pick_log/' + str(j) + '.jpg'
            cv2.imwrite(string, frame)

        counters, frame = detect_contours(frame_filter)

        flag, length, width, angle, s_x, s_y, frame = detect_block(counters, frame_filter)
        if j % 5 == 0:
            print("saved>>>>>")
            string = 'pick_log/' + str(j) + '.jpg'
            cv2.imwrite(string, frame)
        j+=1

        if flag:
            count += 1
            m_angle = (count - 1) / count * m_angle + angle / count
            m_x = (count - 1) / count * m_x + s_x / count
            m_y = (count - 1) / count * m_y + s_y / count
            

        if count == COUNT_MAX:
            print("我的位置",m_x,m_y)
            count = 0
            res = adjust_place(m_angle, m_x, m_y, opposite_yaw=opposite_yaw)
            if res:

                break

    cap.release()
    cv2.destroyAllWindows()


def run(opposite_yaw, color='blue', is_place = False):
    origin_yaw = opposite_yaw
    if is_place:
        seek_place(opposite_yaw, color)
    else:
        while True:
            seek_pick(origin_yaw, color)

            # 读51号舵机【-65， 65】-65是完全打开
            c = dog.read_motor()
            print(c[12])

            if c[12] >0 and c[12]<50:
                break
            
            dog.gait_type("trot")
            adjust_x(-10,1) 

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

if __name__ == '__main__':
    # print(detect_color())
    # input("*"*20)
    # for i in range(5):
    #     oppozite_yaw = dog.read_yaw()
    # left_yaw = oppozite_yaw + 90
    # run(oppozite_yaw, color = 'red', is_place=False)

    for i in range(5):
        oppozite_yaw = dog.read_yaw()
    left_yaw = oppozite_yaw + 90

    run(oppozite_yaw, color='red', is_place=False)