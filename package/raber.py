#!/usr/bin/env python3

import serial  # 用于串行通信
import numpy as np  # 用于数值计算
import cv2  # 用于图像处理
import time  # 用于时间相关操作
import threading  # 用于多线程
import queue  # 用于线程间数据传输
from collections import deque  # 用于高效队列操作，特别是固定长度队列
import math  # 用于数学计算
import argparse  # 用于命令行参数解析


class RadarPositionCorrection:
    """
    雷达位置校正类
    该类处理激光雷达数据，检测后方线条，并计算雷达相对于线条的位置偏移
    """

    def __init__(self,
                 # ========== 可调参数 ==========
                 port='/dev/ttyUSB0',  # 串口设备名
                 baudrate=230400,  # 串口波特率
                 max_range=1200,  # 最大检测范围（毫米）
                 img_size=500,  # 二值图像尺寸（像素）
                 target_distance_from_line=200,  # 距离线条的目标距离（毫米）
                 angle_tolerance=5,  # 角度容差（度）
                 distance_tolerance=20,  # 距离容差（毫米）
                 min_line_length=5,  # 最小线段长度（像素）
                 hough_threshold=30,  # 霍夫变换阈值
                 max_line_gap=10,  # 最大线条间隙（像素）
                 data_collection_time=0.5,  # 实时模式下数据缓冲时间（秒）
                 gaussian_blur_size=1,  # 高斯模糊核尺寸（像素）
                 binary_threshold=30,  # 二值化阈值（0-255）
                 display_scale=1.5,  # 显示缩放比例
                 erosion_kernel_size=1,  # 腐蚀核尺寸
                 erosion_iterations=1,  # 腐蚀迭代次数
                 # ========== 修正后的角度范围参数 ==========
                 rear_angle_min=140,  # 后方角度最小值（度）- 改为检测正后方
                 rear_angle_max=220,  # 后方角度最大值（度）- 改为检测正后方
                 min_rear_distance=50,  # 后方最小距离（毫米）
                 # ==============================
                 ):

        # 雷达参数
        self.port = port
        self.baudrate = baudrate
        self.max_range = max_range

        # 图像处理参数
        self.img_size = img_size
        self.resolution = max_range * 2 / img_size

        # 位置校正参数
        self.target_distance_from_line = target_distance_from_line
        self.angle_tolerance = angle_tolerance
        self.distance_tolerance = distance_tolerance

        # OpenCV参数
        self.min_line_length = min_line_length
        self.hough_threshold = hough_threshold
        self.max_line_gap = max_line_gap
        self.gaussian_blur_size = gaussian_blur_size
        self.binary_threshold = binary_threshold
        self.display_scale = display_scale

        # 形态学操作参数
        self.erosion_kernel_size = erosion_kernel_size
        self.erosion_iterations = erosion_iterations

        # 数据采集参数
        self.data_collection_time = data_collection_time

        # 角度范围参数（修正为正后方）
        self.rear_angle_min = rear_angle_min
        self.rear_angle_max = rear_angle_max
        self.min_rear_distance = min_rear_distance

        # 运行状态
        self.running = False
        self.scan_data = {}
        self.data_queue = queue.Queue(maxsize=200)

        # 实时模式数据
        self.point_buffer = deque(maxlen=1000)
        self.last_update_time = 0
        self.latest_correction = None
        self.visualization_img = None

    def is_angle_in_rear_range(self, angle):
        """检查角度是否在后方范围内"""
        angle = angle % 360
        if self.rear_angle_min <= self.rear_angle_max:
            return self.rear_angle_min <= angle <= self.rear_angle_max
        else:
            return angle >= self.rear_angle_min or angle <= self.rear_angle_max

    def find_frame_header(self, ser):
        """查找数据帧头部 0x54 0x2C"""
        start_time = time.time()
        while self.running and (time.time() - start_time < 5):
            try:
                byte = ser.read(1)
                if byte == b'\x54':
                    next_byte = ser.read(1)
                    if next_byte == b'\x2C':
                        return b'\x54\x2C'
                time.sleep(0.001)
            except:
                return None
        return None

    def parse_frame(self, frame_data):
        """解析47字节数据帧"""
        if len(frame_data) != 47:
            return None

        try:
            speed = frame_data[2] | (frame_data[3] << 8)
            start_angle_raw = frame_data[4] | (frame_data[5] << 8)
            start_angle = start_angle_raw * 0.01
            end_angle_raw = frame_data[42] | (frame_data[43] << 8)
            end_angle = end_angle_raw * 0.01

            points = []
            for i in range(12):
                offset = 6 + i * 3
                distance = frame_data[offset] | (frame_data[offset + 1] << 8)
                intensity = frame_data[offset + 2]
                points.append((distance, intensity))

            return {
                'speed': speed,
                'start_angle': start_angle,
                'end_angle': end_angle,
                'points': points
            }
        except Exception as e:
            print(f"解析帧数据错误: {e}")
            return None

    def calculate_point_angles(self, start_angle, end_angle, num_points=12):
        """计算每个测量点的角度"""
        if end_angle < start_angle:
            end_angle += 360

        angle_step = (end_angle - start_angle) / (num_points - 1)
        angles = []
        for i in range(num_points):
            angle = start_angle + i * angle_step
            angle = angle % 360
            angles.append(angle)
        return angles

    def data_reader_thread(self, ser):
        """数据读取线程"""
        while self.running:
            try:
                header = self.find_frame_header(ser)
                if header is None:
                    continue

                remaining_data = ser.read(45)
                if len(remaining_data) != 45:
                    continue

                frame_data = header + remaining_data
                parsed_data = self.parse_frame(frame_data)
                if parsed_data is None:
                    continue

                if not self.data_queue.full():
                    self.data_queue.put(parsed_data)

            except Exception as e:
                print(f"读取数据错误: {e}")
                time.sleep(0.1)

    def collect_data(self):
        """采集雷达数据"""
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"已连接到 {self.port}")
        except Exception as e:
            print(f"连接串口错误: {e}")
            return False

        self.running = True
        self.scan_data = {}

        reader_thread = threading.Thread(target=self.data_reader_thread, args=(ser,))
        reader_thread.daemon = True
        reader_thread.start()

        start_time = time.time()
        data_count = 0

        print(f"采集数据 {self.data_collection_time} 秒...")
        print(f"后方角度范围: {self.rear_angle_min}° - {self.rear_angle_max}° (正后方)")

        while time.time() - start_time < self.data_collection_time:
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get_nowait()
                    angles = self.calculate_point_angles(data['start_angle'], data['end_angle'])

                    for i, (distance, intensity) in enumerate(data['points']):
                        if 0 < distance <= self.max_range:
                            angle = angles[i]
                            if self.is_angle_in_rear_range(angle):
                                self.scan_data[angle] = (distance, intensity)
                                data_count += 1

                time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"采集数据错误: {e}")
                break

        self.running = False
        ser.close()

        print(f"数据采集完成。采集到 {data_count} 个数据点，{len(self.scan_data)} 个有效角度")
        return len(self.scan_data) > 20

    def process_realtime_data(self):
        """处理实时数据"""
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                angles = self.calculate_point_angles(data['start_angle'], data['end_angle'])
                timestamp = time.time()

                for i, (distance, intensity) in enumerate(data['points']):
                    if 0 < distance <= self.max_range:
                        angle = angles[i]
                        if self.is_angle_in_rear_range(angle):
                            x = distance * np.sin(np.radians(angle))
                            y = distance * np.cos(np.radians(angle))
                            self.point_buffer.append((x, y, timestamp))
                            self.scan_data[angle] = (distance, intensity)
        except queue.Empty:
            pass

        current_time = time.time()
        while self.point_buffer and current_time - self.point_buffer[0][2] > self.data_collection_time:
            self.point_buffer.popleft()

    def world_to_img_coords(self, x, y):
        """世界坐标转图像坐标"""
        img_center = self.img_size // 2
        img_x = int(img_center + x / self.resolution)
        img_y = int(img_center - y / self.resolution)
        img_x = max(0, min(img_x, self.img_size - 1))
        img_y = max(0, min(img_y, self.img_size - 1))
        return img_x, img_y

    def create_binary_image(self, use_buffer=False):
        """创建二值图像"""
        binary_img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        if use_buffer:
            for x, y, _ in self.point_buffer:
                img_x, img_y = self.world_to_img_coords(x, y)
                cv2.circle(binary_img, (img_x, img_y), 1, 255, -1)
        else:
            for angle, (distance, intensity) in self.scan_data.items():
                if distance <= self.max_range:
                    x = distance * np.sin(np.radians(angle))
                    y = distance * np.cos(np.radians(angle))
                    img_x, img_y = self.world_to_img_coords(x, y)
                    cv2.circle(binary_img, (img_x, img_y), 1, 255, -1)

        binary_img = cv2.GaussianBlur(binary_img, (self.gaussian_blur_size, self.gaussian_blur_size), 0)
        _, binary_img = cv2.threshold(binary_img, self.binary_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.erosion_kernel_size, self.erosion_kernel_size))
        binary_img = cv2.erode(binary_img, kernel, iterations=self.erosion_iterations)

        return binary_img

    def detect_rear_line(self, use_buffer=False):
        """检测后方线条"""
        binary_img = self.create_binary_image(use_buffer)

        lines = cv2.HoughLinesP(
            binary_img,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None or len(lines) == 0:
            return None, binary_img

        rear_lines = []
        center = self.img_size // 2

        for line in lines:
            x1, y1, x2, y2 = line[0]
            wx1 = (x1 - center) * self.resolution
            wy1 = (center - y1) * self.resolution
            wx2 = (x2 - center) * self.resolution
            wy2 = (center - y2) * self.resolution

            mid_x = (wx1 + wx2) / 2
            mid_y = (wy1 + wy2) / 2

            angle_to_line = math.degrees(math.atan2(mid_x, mid_y))
            if angle_to_line < 0:
                angle_to_line += 360

            distance_to_radar = math.sqrt(mid_x ** 2 + mid_y ** 2)

            if (self.is_angle_in_rear_range(angle_to_line) and
                    distance_to_radar > self.min_rear_distance):
                length = math.sqrt((wx2 - wx1) ** 2 + (wy2 - wy1) ** 2)
                line_angle = math.degrees(math.atan2(wy2 - wy1, wx2 - wx1))

                rear_lines.append({
                    'x1': wx1, 'y1': wy1, 'x2': wx2, 'y2': wy2,
                    'img_x1': x1, 'img_y1': y1, 'img_x2': x2, 'img_y2': y2,
                    'length': length,
                    'angle': line_angle,
                    'center_angle': angle_to_line,
                    'center_distance': distance_to_radar
                })

        if not rear_lines:
            return None, binary_img

        rear_lines.sort(key=lambda x: x['length'], reverse=True)
        return rear_lines[0], binary_img

    def calculate_position_correction(self, line):
        """计算位置校正参数"""
        if line is None:
            return None

        mid_x = (line['x1'] + line['x2']) / 2
        mid_y = (line['y1'] + line['y2']) / 2

        angle_with_x = line['angle']
        if angle_with_x > 90:
            angle_with_x -= 180
        elif angle_with_x < -90:
            angle_with_x += 180

        A = line['y2'] - line['y1']
        B = line['x1'] - line['x2']
        C = line['x2'] * line['y1'] - line['x1'] * line['y2']
        distance_to_line = abs(C) / math.sqrt(A * A + B * B)

        horizontal_offset = -mid_x
        current_vertical_distance = distance_to_line
        vertical_adjustment = self.target_distance_from_line - current_vertical_distance

        return {
            'horizontal_offset': horizontal_offset,
            'vertical_offset': vertical_adjustment,
            'angle_offset': angle_with_x,
            'line_center_x': mid_x,
            'line_center_y': mid_y,
            'distance_to_line': distance_to_line,
            'line_length': line['length'],
            'center_angle': line.get('center_angle', 0),
            'is_position_ok': (abs(horizontal_offset) < self.distance_tolerance and
                               abs(vertical_adjustment) < self.distance_tolerance and
                               abs(angle_with_x) < self.angle_tolerance)
        }

    def draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=10):
        """绘制虚线"""
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        dashes = int(dist / (dash_length + gap_length))

        if dashes == 0:
            cv2.line(img, pt1, pt2, color, thickness)
            return

        for i in range(dashes):
            start_factor = i * (dash_length + gap_length) / dist
            end_factor = min(1, (i * (dash_length + gap_length) + dash_length) / dist)

            start_pt = (int(pt1[0] + (pt2[0] - pt1[0]) * start_factor),
                        int(pt1[1] + (pt2[1] - pt1[1]) * start_factor))
            end_pt = (int(pt1[0] + (pt2[0] - pt1[0]) * end_factor),
                      int(pt1[1] + (pt2[1] - pt1[1]) * end_factor))

            cv2.line(img, start_pt, end_pt, color, thickness)

    def create_visualization(self, binary_img, line, correction):
        """创建可视化图像"""
        vis_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        center = self.img_size // 2

        # 绘制坐标轴
        cv2.line(vis_img, (0, center), (self.img_size, center), (0, 255, 0), 1)
        cv2.line(vis_img, (center, 0), (center, self.img_size), (0, 255, 0), 1)

        # 绘制雷达位置
        cv2.circle(vis_img, (center, center), 5, (0, 0, 255), -1)

        # 绘制角度范围指示线
        for angle in [self.rear_angle_min, self.rear_angle_max]:
            end_x = int(center + (self.img_size // 3) * np.sin(np.radians(angle)))
            end_y = int(center - (self.img_size // 3) * np.cos(np.radians(angle)))
            cv2.line(vis_img, (center, center), (end_x, end_y), (128, 128, 128), 1)

        if line is not None:
            # 绘制检测到的线条
            cv2.line(vis_img, (line['img_x1'], line['img_y1']),
                     (line['img_x2'], line['img_y2']), (255, 0, 0), 2)

            if correction is not None:
                # 绘制相关线条和信息
                mid_x = (line['img_x1'] + line['img_x2']) // 2
                mid_y = (line['img_y1'] + line['img_y2']) // 2

                dx = line['img_x2'] - line['img_x1']
                dy = line['img_y2'] - line['img_y1']
                length = math.sqrt(dx * dx + dy * dy)

                if length > 0:
                    nx = -dy / length
                    ny = dx / length

                    target_dist_px = self.target_distance_from_line / self.resolution
                    target_x1 = int(line['img_x1'] + nx * target_dist_px)
                    target_y1 = int(line['img_y1'] + ny * target_dist_px)
                    target_x2 = int(line['img_x2'] + nx * target_dist_px)
                    target_y2 = int(line['img_y2'] + ny * target_dist_px)

                    self.draw_dashed_line(vis_img, (target_x1, target_y1),
                                          (target_x2, target_y2), (0, 255, 255), 1)

                    end_x = int(center + nx * (correction['distance_to_line'] / self.resolution))
                    end_y = int(center + ny * (correction['distance_to_line'] / self.resolution))
                    cv2.line(vis_img, (center, center), (end_x, end_y), (255, 255, 0), 2)

        # 添加信息文本
        info_lines = []
        if correction is not None:
            h_offset = correction['horizontal_offset']
            v_offset = correction['vertical_offset']
            a_offset = correction['angle_offset']
            status = "✓ Position OK" if correction['is_position_ok'] else "✗ Needs Adjustment"

            info_lines = [
                f"Rear Detection Range: {self.rear_angle_min}°-{self.rear_angle_max}° (Directly behind)",
                f"Horizontal: {h_offset:.1f} mm {'←' if h_offset > 0 else '→' if h_offset < 0 else ''}",
                f"Vertical: {v_offset:.1f} mm {'↑' if v_offset > 0 else '↓' if v_offset < 0 else ''}",
                f"Angle: {a_offset:.1f}° {'↺' if a_offset > 0 else '↻' if a_offset < 0 else ''}",
                f"Distance: {correction['distance_to_line']:.1f} mm",
                f"Target: {self.target_distance_from_line} mm",
                f"Line Angle: {correction.get('center_angle', 0):.1f}°",
                status
            ]
        else:
            info_lines = [
                f"Rear Detection Range: {self.rear_angle_min}°-{self.rear_angle_max}° (Directly behind)",
                "No rear line detected"
            ]

        for i, line_text in enumerate(info_lines):
            y = 30 + i * 25
            cv2.putText(vis_img, line_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_img, line_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        return vis_img

    def realtime_detection_thread(self):
        """实时检测线程"""
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"已连接到 {self.port}")
        except Exception as e:
            print(f"连接串口错误: {e}")
            return

        self.running = True
        self.scan_data = {}
        self.point_buffer.clear()

        reader_thread = threading.Thread(target=self.data_reader_thread, args=(ser,))
        reader_thread.daemon = True
        reader_thread.start()

        print(f"开始实时检测... 后方角度范围: {self.rear_angle_min}°-{self.rear_angle_max}° (正后方)")

        try:
            while self.running:
                self.process_realtime_data()

                current_time = time.time()
                if current_time - self.last_update_time >= 0.5 and len(self.point_buffer) >= 20:
                    line, binary_img = self.detect_rear_line(use_buffer=True)

                    if line is not None:
                        self.latest_correction = self.calculate_position_correction(line)
                        self.visualization_img = self.create_visualization(binary_img, line, self.latest_correction)
                    else:
                        self.latest_correction = None
                        self.visualization_img = self.create_visualization(binary_img, None, None)

                    self.last_update_time = current_time

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("用户中断检测")
        except Exception as e:
            print(f"实时检测错误: {e}")
        finally:
            self.running = False
            ser.close()
            print("实时检测已停止")

    def start_realtime_detection(self):
        """启动实时检测"""
        self.detection_thread = threading.Thread(target=self.realtime_detection_thread)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        cv2.namedWindow("雷达位置检测", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("雷达位置检测", int(self.img_size * self.display_scale),
                         int(self.img_size * self.display_scale))
        try:
            while self.running:
                if self.visualization_img is not None:
                    cv2.imshow("雷达位置检测", self.visualization_img)

                key = cv2.waitKey(10)
                if key == 27 or key == ord('q'):
                    break

        except KeyboardInterrupt:
            print("用户中断显示")
        finally:
            self.running = False
            cv2.destroyAllWindows()
            if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1.0)


def run_radar_realtime(port='/dev/ttyUSB0',
                       baudrate=230400,
                       max_range=1200,
                       target_distance=200,
                       rear_angle_min=160,  # 修改为正后方
                       rear_angle_max=200):  # 修改为正后方
    """运行激光雷达实时位置检测"""
    radar = RadarPositionCorrection(
        port=port,
        baudrate=baudrate,
        max_range=max_range,
        target_distance_from_line=target_distance,
        rear_angle_min=rear_angle_min,
        rear_angle_max=rear_angle_max
    )

    try:
        radar.start_realtime_detection()
    except Exception as e:
        print(f"实时检测错误: {e}")

def run_radar(port='/dev/ttyUSB0',
              baudrate=230400,
              max_range=1200,
              target_distance=200,
              collection_time=2.0,
              rear_angle_min=160,  # 修改为正后方
              rear_angle_max=200): # 修改为正后方
    """运行激光雷达单次位置检测"""
    radar = RadarPositionCorrection(
        port=port,
        baudrate=baudrate,
        max_range=max_range,
        target_distance_from_line=target_distance,
        data_collection_time=collection_time,
        rear_angle_min=rear_angle_min,
        rear_angle_max=rear_angle_max
    )

    try:
        if not radar.collect_data():
            return {
                'success': False,
                'error': '采集雷达数据失败',
                'horizontal_offset': 0,
                'vertical_offset': 0,
                'angle_offset': 0,
                'distance_to_line': 0,
                'is_position_ok': False,
                'line_info': None
            }

        rear_line, _ = radar.detect_rear_line()

        if rear_line is None:
            return {
                'success': False,
                'error': f'未检测到后方线条（角度范围：{rear_angle_min}°-{rear_angle_max}°，正后方）',
                'horizontal_offset': 0,
                'vertical_offset': 0,
                'angle_offset': 0,
                'distance_to_line': 0,
                'is_position_ok': False,
                'line_info': None
            }

        correction = radar.calculate_position_correction(rear_line)

        if correction is None:
            return {
                'success': False,
                'error': '计算位置校正参数失败',
                'horizontal_offset': 0,
                'vertical_offset': 0,
                'angle_offset': 0,
                'distance_to_line': 0,
                'is_position_ok': False,
                'line_info': None
            }

        print(f"位置检测结果（角度范围：{rear_angle_min}°-{rear_angle_max}°，正后方）:")
        print(f"  水平偏移: {correction['horizontal_offset']:.1f} 毫米")
        print(f"  垂直偏移: {correction['vertical_offset']:.1f} 毫米")
        print(f"  角度偏移: {correction['angle_offset']:.1f} 度")
        print(f"  到线条距离: {correction['distance_to_line']:.1f} 毫米")
        print(f"  线条位置角度: {correction.get('center_angle', 0):.1f} 度")
        print(f"  位置状态: {'正确' if correction['is_position_ok'] else '需要调整'}")

        return {
            'success': True,
            'horizontal_offset': correction['horizontal_offset'],
            'vertical_offset': correction['vertical_offset'],
            'angle_offset': correction['angle_offset'],
            'distance_to_line': correction['distance_to_line'],
            'is_position_ok': correction['is_position_ok'],
            'line_info': rear_line
        }

    except Exception as e:
        print(f"雷达检测错误: {e}")
        return {
            'success': False,
            'error': str(e),
            'horizontal_offset': 0,
            'vertical_offset': 0,
            'angle_offset': 0,
            'distance_to_line': 0,
            'is_position_ok': False,
            'line_info': None
        }


def run_distance(aim_distance, max_line_distance,angle_min=140,angle_max=220):
    """
    根据目标距离和最大线条距离运行雷达检测

    参数:
        aim_distance: 目标距离（毫米）
        max_line_distance: 最大线条距离（毫米）

    返回值:
        dict: 检测结果
    """
    result=run_radar(
    port='/dev/ttyUSB0',
    baudrate=230400,
    max_range=max_line_distance,
    target_distance=aim_distance,
    collection_time=2.0,
    rear_angle_min=angle_min,  # 修改为正后方
    rear_angle_max=angle_max   # 修改为正后方
)
    if result['success']:
        print("\n=== 检测成功 ===")
        print(f"水平偏移: {result['horizontal_offset']:.1f} 毫米")
        print(f"垂直偏移: {result['vertical_offset']:.1f} 毫米")
        print(f"角度偏移: {result['angle_offset']:.1f} 度")
        print(f"距离线条: {result['distance_to_line']:.1f} 毫米")
        print(f"位置状态: {'正确' if result['is_position_ok'] else '需要调整'}")
        return result['distance_to_line'],result['angle_offset']
    else:
        return None,None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='激光雷达位置校正系统')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='串口设备名')
    parser.add_argument('--baudrate', type=int, default=230400, help='串口波特率')
    parser.add_argument('--max-range', type=int, default=1200, help='最大检测范围（毫米）')
    parser.add_argument('--target-distance', type=int, default=200, help='目标距离（毫米）')
    parser.add_argument('--rear-angle-min', type=int, default=160, help='后方角度最小值（度）- 正后方')
    parser.add_argument('--rear-angle-max', type=int, default=200, help='后方角度最大值（度）- 正后方')
    parser.add_argument('--mode', choices=['single', 'realtime'], default='realtime',
                       help='运行模式：single（单次检测）或 realtime（实时检测）')
    parser.add_argument('--collection-time', type=float, default=2.0,
                       help='单次模式数据采集时间（秒）')

    args = parser.parse_args()

    print(f"启动{'实时' if args.mode == 'realtime' else '单次'}雷达位置检测...")
    print(f"后方检测角度范围: {args.rear_angle_min}°-{args.rear_angle_max}° (正后方)")
    print("按ESC或q键退出")

    try:
        if args.mode == 'single':
            # 实时模式
            run_radar_realtime(
                port=args.port,
                baudrate=args.baudrate,
                max_range=args.max_range,
                target_distance=args.target_distance,
                rear_angle_min=args.rear_angle_min,
                rear_angle_max=args.rear_angle_max
            )
        else:
            # 单次模式
            result = run_radar(
                port=args.port,
                baudrate=args.baudrate,
                max_range=args.max_range,
                target_distance=args.target_distance,
                collection_time=args.collection_time,
                rear_angle_min=args.rear_angle_min,
                rear_angle_max=args.rear_angle_max
            )

            if result['success']:
                print("\n=== 检测成功 ===")
                print(f"水平偏移: {result['horizontal_offset']:.1f} 毫米")
                print(f"垂直偏移: {result['vertical_offset']:.1f} 毫米")
                print(f"角度偏移: {result['angle_offset']:.1f} 度")
                print(f"距离线条: {result['distance_to_line']:.1f} 毫米")
                print(f"位置状态: {'正确' if result['is_position_ok'] else '需要调整'}")
            else:
                print(f"\n=== 检测失败 ===")
                print(f"错误: {result['error']}")

    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序错误: {e}")


if __name__ == "__main__":
    main()
    #a,b=run_distance(200,1200)