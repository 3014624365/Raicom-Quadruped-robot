#!/usr/bin/env python3

import serial  # For serial communication
import numpy as np  # For numerical computation
import time  # For time-related operations
import threading  # For multi-threading
import queue  # For inter-thread data transfer
from collections import deque  # For efficient queue operations, especially fixed-length queues
import math  # For mathematical calculations
import argparse  # For command line argument parsing


class RadarPositionCorrection:
    """
    Radar Position Correction Class
    This class processes laser radar data, detects rear lines, and calculates radar position offset relative to the line
    """

    def __init__(self,
                 # ========== Adjustable Parameters ==========
                 port='/dev/ttyUSB0',  # Serial port device name
                 baudrate=230400,  # Serial port baud rate
                 max_range=800,  # Maximum detection range (mm)
                 img_size=500,  # Binary image size (pixels)
                 target_distance_from_line=200,  # Target distance from line (mm)
                 angle_tolerance=5,  # Angle tolerance (degrees)
                 distance_tolerance=20,  # Distance tolerance (mm)
                 min_line_length=10,  # Minimum line length (pixels)
                 hough_threshold=40,  # Hough transform threshold
                 max_line_gap=10,  # Maximum line gap (pixels)
                 data_collection_time=0.5,  # Data buffer time in real-time mode (seconds)
                 gaussian_blur_size=1,  # Gaussian blur kernel size (pixels)
                 binary_threshold=30,  # Binary threshold (0-255)
                 erosion_kernel_size=2,  # Erosion kernel size
                 erosion_iterations=1,  # Erosion iterations
                 # ==============================
                 ):

        # Radar parameters
        self.port = port  # Serial port device name
        self.baudrate = baudrate  # Serial port baud rate
        self.max_range = max_range  # Maximum detection range (mm)

        # Image processing parameters
        self.img_size = img_size  # Image size (pixels)
        self.resolution = max_range * 2 / img_size  # Actual distance per pixel (mm/pixel)

        # Position correction parameters
        self.target_distance_from_line = target_distance_from_line  # Target distance (mm)
        self.angle_tolerance = angle_tolerance  # Angle tolerance (degrees)
        self.distance_tolerance = distance_tolerance  # Distance tolerance (mm)

        # OpenCV parameters
        self.min_line_length = min_line_length  # Minimum line segment length for Hough transform
        self.hough_threshold = hough_threshold  # Hough transform threshold
        self.max_line_gap = max_line_gap  # Maximum line gap
        self.gaussian_blur_size = gaussian_blur_size  # Gaussian blur kernel size
        self.binary_threshold = binary_threshold  # Binary threshold

        # Morphological operation parameters
        self.erosion_kernel_size = erosion_kernel_size
        self.erosion_iterations = erosion_iterations

        # Data collection parameters
        self.data_collection_time = data_collection_time  # Data collection time (seconds)

        # Running status
        self.running = False  # Running flag
        self.scan_data = {}  # Scan data, format: {angle: (distance, intensity)}
        self.data_queue = queue.Queue(maxsize=200)  # Data queue for inter-thread communication

        # Real-time mode data
        self.point_buffer = deque(maxlen=1000)  # Point buffer, stores recent point data, format: (x, y, timestamp)
        self.last_update_time = 0  # Last update time
        self.latest_correction = None  # Latest position correction result

    def find_frame_header(self, ser):
        """
        Find data frame header 0x54 0x2C

        Args:
            ser: Serial port object

        Returns:
            bytes: Found frame header, returns None if not found
        """
        start_time = time.time()
        # Set 5 second timeout
        while self.running and (time.time() - start_time < 5):
            try:
                byte = ser.read(1)  # Read one byte
                if byte == b'\x54':  # Check if first byte is 0x54
                    next_byte = ser.read(1)  # Read next byte
                    if next_byte == b'\x2C':  # Check if next byte is 0x2C
                        return b'\x54\x2C'  # Return complete frame header
                time.sleep(0.001)  # Brief sleep to avoid high CPU usage
            except:
                return None  # Return None on exception
        return None  # Return None on timeout

    def parse_frame(self, frame_data):
        """
        Parse 47-byte data frame

        Args:
            frame_data: 47-byte frame data

        Returns:
            dict: Parsed data including speed, angle and point cloud data; returns None if parsing fails
        """
        if len(frame_data) != 47:  # Check data length
            return None

        try:
            # Parse motor speed
            speed = frame_data[2] | (frame_data[3] << 8)

            # Parse start angle (unit: 0.01 degrees)
            start_angle_raw = frame_data[4] | (frame_data[5] << 8)
            start_angle = start_angle_raw * 0.01  # Convert to degrees

            # Parse end angle (unit: 0.01 degrees)
            end_angle_raw = frame_data[42] | (frame_data[43] << 8)
            end_angle = end_angle_raw * 0.01  # Convert to degrees

            # Parse distance and intensity data
            points = []
            for i in range(12):  # Each frame contains 12 points
                offset = 6 + i * 3
                # Distance is 2 bytes, low byte first
                distance = frame_data[offset] | (frame_data[offset + 1] << 8)
                # Intensity is 1 byte
                intensity = frame_data[offset + 2]
                points.append((distance, intensity))

            # Return parsed data
            return {
                'speed': speed,  # Motor speed
                'start_angle': start_angle,  # Start angle
                'end_angle': end_angle,  # End angle
                'points': points  # Point list: [(distance, intensity), ...]
            }

        except Exception as e:
            print(f"Error parsing frame data: {e}")
            return None

    def calculate_point_angles(self, start_angle, end_angle, num_points=12):
        """
        Calculate angle for each measurement point

        Args:
            start_angle: Start angle (degrees)
            end_angle: End angle (degrees)
            num_points: Number of points, default 12

        Returns:
            list: List of angles for each point
        """
        # If end angle is less than start angle, it means crossing 0 degrees
        if end_angle < start_angle:
            end_angle += 360

        # Calculate angle step
        angle_step = (end_angle - start_angle) / (num_points - 1)

        # Calculate angle for each point
        angles = []
        for i in range(num_points):
            angle = start_angle + i * angle_step
            angle = angle % 360  # Ensure angle is in 0-360 range
            angles.append(angle)
        return angles

    def data_reader_thread(self, ser):
        """
        Data reading thread, continuously reads data from serial port and puts into queue

        Args:
            ser: Serial port object
        """
        while self.running:
            try:
                # Find frame header
                header = self.find_frame_header(ser)
                if header is None:
                    continue

                # Read remaining frame data
                remaining_data = ser.read(45)  # Frame header 2 bytes + remaining 45 bytes = 47 bytes
                if len(remaining_data) != 45:
                    continue

                # Combine complete frame data
                frame_data = header + remaining_data

                # Parse frame data
                parsed_data = self.parse_frame(frame_data)
                if parsed_data is None:
                    continue

                # Put parsed data into queue
                if not self.data_queue.full():
                    self.data_queue.put(parsed_data)

            except Exception as e:
                print(f"Error reading data: {e}")
                time.sleep(0.1)  # Brief sleep on error

    def collect_data(self):
        """
        Collect radar data

        Returns:
            bool: Returns True if collection successful, otherwise False
        """
        try:
            # Open serial port
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.port}")
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            return False

        self.running = True
        self.scan_data = {}

        # Start data reading thread
        reader_thread = threading.Thread(target=self.data_reader_thread, args=(ser,))
        reader_thread.daemon = True  # Set as daemon thread, automatically ends when main thread ends
        reader_thread.start()

        # Collect data
        start_time = time.time()
        data_count = 0

        print(f"Collecting data for {self.data_collection_time} seconds...")

        # Collect data within specified time
        while time.time() - start_time < self.data_collection_time:
            try:
                # Get data from queue
                if not self.data_queue.empty():
                    data = self.data_queue.get_nowait()

                    # Calculate angle for each point
                    angles = self.calculate_point_angles(data['start_angle'], data['end_angle'])

                    # Process each point's data
                    for i, (distance, intensity) in enumerate(data['points']):
                        # Filter invalid points and out-of-range points
                        if 0 < distance <= self.max_range:
                            angle = angles[i]
                            self.scan_data[angle] = (
                                distance, intensity)  # Store distance and intensity with angle as key
                            data_count += 1

                time.sleep(0.01)  # Brief sleep
            except queue.Empty:
                continue  # Queue empty, continue loop
            except Exception as e:
                print(f"Error collecting data: {e}")
                break

        self.running = False
        ser.close()  # Close serial port

        print(f"Data collection complete. Collected {data_count} data points, {len(self.scan_data)} valid angles")
        return len(self.scan_data) > 50  # Need at least 50 valid points to consider collection successful

    def process_realtime_data(self):
        """
        Process real-time data, read data from queue and update point buffer
        """
        try:
            # Process all data in queue
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()

                # Calculate angle for each point
                angles = self.calculate_point_angles(data['start_angle'], data['end_angle'])

                # Current timestamp
                timestamp = time.time()

                # Process each point
                for i, (distance, intensity) in enumerate(data['points']):
                    # Filter invalid points and out-of-range points
                    if 0 < distance <= self.max_range:
                        angle = angles[i]

                        # Convert polar coordinates to Cartesian coordinates
                        x = distance * np.sin(np.radians(angle))  # X-axis direction (horizontal)
                        y = distance * np.cos(np.radians(angle))  # Y-axis direction (vertical)

                        # Add point data to buffer
                        self.point_buffer.append((x, y, timestamp))

                        # Update scan data
                        self.scan_data[angle] = (distance, intensity)
        except queue.Empty:
            pass  # Queue empty, no processing needed

        # Clear expired data points (points older than data collection time)
        current_time = time.time()
        while self.point_buffer and current_time - self.point_buffer[0][2] > self.data_collection_time:
            self.point_buffer.popleft()  # Remove from left side (oldest data)

    def world_to_img_coords(self, x, y):
        """
        Convert world coordinates (millimeters) to image coordinates (pixels)

        Args:
            x: X coordinate (millimeters)
            y: Y coordinate (millimeters)

        Returns:
            tuple: (img_x, img_y) image coordinates
        """
        # Image center point (radar position)
        img_center = self.img_size // 2

        # Coordinate conversion (note y-axis direction is reversed)
        img_x = int(img_center + x / self.resolution)  # Right is positive
        img_y = int(img_center - y / self.resolution)  # Up is positive

        # Limit within image range
        img_x = max(0, min(img_x, self.img_size - 1))
        img_y = max(0, min(img_y, self.img_size - 1))

        return img_x, img_y

    def create_binary_image(self, use_buffer=False):
        """
        Create binary image showing radar point cloud data

        Args:
            use_buffer: Whether to use point buffer data, True for buffer, False for scan data

        Returns:
            ndarray: Binary image
        """
        try:
            import cv2
        except ImportError:
            # 如果没有OpenCV，创建一个模拟的二值图像数据结构
            return self.create_mock_binary_image(use_buffer)

        # Create blank image
        binary_img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        if use_buffer:
            # Use point buffer data (real-time mode)
            for x, y, _ in self.point_buffer:
                img_x, img_y = self.world_to_img_coords(x, y)
                cv2.circle(binary_img, (img_x, img_y), 1, 255, -1)  # Draw point (radius 1 pixel)
        else:
            # Use scan data (single mode)
            for angle, (distance, intensity) in self.scan_data.items():
                if distance <= self.max_range:
                    # Polar to Cartesian coordinates
                    x = distance * np.sin(np.radians(angle))
                    y = distance * np.cos(np.radians(angle))

                    # World to image coordinates
                    img_x, img_y = self.world_to_img_coords(x, y)

                    # Draw point
                    cv2.circle(binary_img, (img_x, img_y), 1, 255, -1)

        # Light Gaussian blur to smooth point cloud
        binary_img = cv2.GaussianBlur(binary_img, (self.gaussian_blur_size, self.gaussian_blur_size), 0)

        # Binary processing to remove noise
        _, binary_img = cv2.threshold(binary_img, self.binary_threshold, 255, cv2.THRESH_BINARY)

        # Erosion operation to thin lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.erosion_kernel_size, self.erosion_kernel_size))
        binary_img = cv2.erode(binary_img, kernel, iterations=self.erosion_iterations)

        return binary_img

    def create_mock_binary_image(self, use_buffer=False):
        """
        创建模拟的二值图像（当OpenCV不可用时）
        """
        # 创建一个简单的点集合来替代图像处理
        points = []

        if use_buffer:
            # 使用点缓冲区数据
            for x, y, _ in self.point_buffer:
                img_x, img_y = self.world_to_img_coords(x, y)
                points.append((img_x, img_y))
        else:
            # 使用扫描数据
            for angle, (distance, intensity) in self.scan_data.items():
                if distance <= self.max_range:
                    x = distance * np.sin(np.radians(angle))
                    y = distance * np.cos(np.radians(angle))
                    img_x, img_y = self.world_to_img_coords(x, y)
                    points.append((img_x, img_y))

        return {'type': 'mock', 'points': points, 'size': self.img_size}

    def detect_rear_line(self, use_buffer=False):
        """
        Detect rear line

        Args:
            use_buffer: Whether to use point buffer data

        Returns:
            tuple: (detected line info dict, binary image)
                   Returns (None, binary image) if no line detected
        """
        # Create binary image
        binary_img = self.create_binary_image(use_buffer)

        # 如果是模拟图像，使用简化的线检测
        if isinstance(binary_img, dict) and binary_img.get('type') == 'mock':
            return self.detect_rear_line_mock(binary_img), binary_img

        try:
            import cv2

            # Use Hough transform to detect lines
            lines = cv2.HoughLinesP(
                binary_img,  # Input image
                rho=1,  # Distance resolution (pixels)
                theta=np.pi / 180,  # Angle resolution (radians)
                threshold=self.hough_threshold,  # Voting threshold
                minLineLength=self.min_line_length,  # Minimum line length
                maxLineGap=self.max_line_gap  # Maximum gap
            )

            # Check if lines detected
            if lines is None or len(lines) == 0:
                return None, binary_img

            # Filter rear lines (lines with negative y coordinates, i.e., behind radar)
            rear_lines = []
            center = self.img_size // 2  # Image center (radar position)

            for line in lines:
                x1, y1, x2, y2 = line[0]  # Line endpoint image coordinates

                # Convert to world coordinates
                wx1 = (x1 - center) * self.resolution  # X coordinate (millimeters)
                wy1 = (center - y1) * self.resolution  # Y coordinate (millimeters)
                wx2 = (x2 - center) * self.resolution
                wy2 = (center - y2) * self.resolution

                # Check if rear line (both endpoints have negative y coordinates, i.e., behind radar)
                if wy1 < -50 and wy2 < -50:  # At least 50mm from radar
                    # Calculate line length
                    length = math.sqrt((wx2 - wx1) ** 2 + (wy2 - wy1) ** 2)

                    # Calculate angle with x-axis
                    angle = math.degrees(math.atan2(wy2 - wy1, wx2 - wx1))

                    # Save line info including world coordinates, image coordinates, length and angle
                    rear_lines.append({
                        'x1': wx1, 'y1': wy1, 'x2': wx2, 'y2': wy2,  # World coordinates (millimeters)
                        'img_x1': x1, 'img_y1': y1, 'img_x2': x2, 'img_y2': y2,  # Image coordinates (pixels)
                        'length': length,  # Length (millimeters)
                        'angle': angle  # Angle (degrees)
                    })

            # Check if rear lines exist
            if not rear_lines:
                return None, binary_img

            # Select longest rear line (more reliable)
            rear_lines.sort(key=lambda x: x['length'], reverse=True)
            return rear_lines[0], binary_img  # Return longest line and binary image

        except ImportError:
            # OpenCV不可用时的简化线检测
            return self.detect_rear_line_mock(binary_img), binary_img

    def detect_rear_line_mock(self, mock_img):
        """
        简化的线检测（当OpenCV不可用时）
        """
        if not mock_img.get('points'):
            return None

        points = mock_img['points']
        center = self.img_size // 2

        # 筛选后方的点（y坐标大于中心的点）
        rear_points = []
        for img_x, img_y in points:
            # 转换为世界坐标检查是否在后方
            wx = (img_x - center) * self.resolution
            wy = (center - img_y) * self.resolution
            if wy < -50:  # 后方至少50mm
                rear_points.append((img_x, img_y, wx, wy))

        if len(rear_points) < 2:
            return None

        # 简单的线拟合：使用最左和最右的点
        rear_points.sort(key=lambda p: p[0])  # 按x坐标排序

        if len(rear_points) >= 2:
            # 使用首尾两点作为线段
            p1 = rear_points[0]
            p2 = rear_points[-1]

            # 计算线段长度
            length = math.sqrt((p2[2] - p1[2]) ** 2 + (p2[3] - p1[3]) ** 2)

            # 计算角度
            angle = math.degrees(math.atan2(p2[3] - p1[3], p2[2] - p1[2]))

            return {
                'x1': p1[2], 'y1': p1[3], 'x2': p2[2], 'y2': p2[3],  # 世界坐标
                'img_x1': p1[0], 'img_y1': p1[1], 'img_x2': p2[0], 'img_y2': p2[1],  # 图像坐标
                'length': length,
                'angle': angle
            }

        return None

    def calculate_position_correction(self, line):
        """
        Calculate position correction parameters

        Args:
            line: Detected line information

        Returns:
            dict: Position correction parameters including horizontal offset, vertical offset, angle offset etc.
                 Returns None if line is None
        """
        if line is None:
            return None

        # Calculate line midpoint
        mid_x = (line['x1'] + line['x2']) / 2
        mid_y = (line['y1'] + line['y2']) / 2

        # Calculate angle with x-axis (normalized to -90 to 90 degrees)
        angle_with_x = line['angle']
        if angle_with_x > 90:
            angle_with_x -= 180
        elif angle_with_x < -90:
            angle_with_x += 180

        # Calculate distance from radar position (0,0) to line
        # Using point-to-line distance formula: |Ax + By + C|/sqrt(A^2 + B^2)
        A = line['y2'] - line['y1']  # A in general form Ax + By + C = 0
        B = line['x1'] - line['x2']  # B
        C = line['x2'] * line['y1'] - line['x1'] * line['y2']  # C

        # Distance from point (0,0) to line
        distance_to_line = abs(C) / math.sqrt(A * A + B * B)

        # Calculate required position adjustment
        # Horizontal offset: negative of line midpoint x coordinate (negative means left, positive means right)
        horizontal_offset = -mid_x

        # Vertical adjustment: difference between target distance and current distance (positive means forward, negative means backward)
        current_vertical_distance = distance_to_line
        vertical_adjustment = self.target_distance_from_line - current_vertical_distance

        # Return position correction parameters
        return {
            'horizontal_offset': horizontal_offset,  # Horizontal offset (mm)
            'vertical_offset': vertical_adjustment,  # Vertical offset (mm)
            'angle_offset': angle_with_x,  # Angle offset (degrees)
            'line_center_x': mid_x,  # Line center x coordinate
            'line_center_y': mid_y,  # Line center y coordinate
            'distance_to_line': distance_to_line,  # Current distance to line
            'line_length': line['length'],  # Line length
            'is_position_ok': (abs(horizontal_offset) < self.distance_tolerance and
                               abs(vertical_adjustment) < self.distance_tolerance and
                               abs(angle_with_x) < self.angle_tolerance)  # Whether position is within tolerance
        }

    def realtime_detection_thread(self):
        """
        Real-time detection thread, continuously processes radar data and updates position correction
        """
        try:
            # Open serial port
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.port}")
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            return

        # Initialize
        self.running = True
        self.scan_data = {}
        self.point_buffer.clear()

        # Start data reading thread
        reader_thread = threading.Thread(target=self.data_reader_thread, args=(ser,))
        reader_thread.daemon = True
        reader_thread.start()

        print("Starting real-time detection...")
        print("Press Ctrl+C to stop")

        try:
            while self.running:
                # Process data in queue
                self.process_realtime_data()

                # Periodically update position correction calculation (every 0.5 seconds)
                current_time = time.time()
                if current_time - self.last_update_time >= 0.5 and len(self.point_buffer) >= 50:
                    # Detect rear line
                    line, binary_img = self.detect_rear_line(use_buffer=True)

                    if line is not None:
                        # Calculate position correction
                        self.latest_correction = self.calculate_position_correction(line)

                        # Print real-time distance information
                        if self.latest_correction:
                            print(f"实际距离: {self.latest_correction['distance_to_line']:.1f} mm | "
                                  f"目标距离: {self.target_distance_from_line} mm | "
                                  f"水平偏移: {self.latest_correction['horizontal_offset']:.1f} mm | "
                                  f"垂直偏移: {self.latest_correction['vertical_offset']:.1f} mm | "
                                  f"角度偏移: {self.latest_correction['angle_offset']:.1f}° | "
                                  f"状态: {'正常' if self.latest_correction['is_position_ok'] else '需要调整'}")
                    else:
                        self.latest_correction = None
                        print("未检测到后方线条")

                    # Update last update time
                    self.last_update_time = current_time

                time.sleep(0.01)  # Brief sleep to avoid high CPU usage

        except KeyboardInterrupt:
            print("\n用户中断检测")
        except Exception as e:
            print(f"Real-time detection error: {e}")
        finally:
            # Clean up resources
            self.running = False
            ser.close()
            print("实时检测已停止")

    def start_realtime_detection(self):
        """
        Start real-time detection (without visualization)
        """
        # Create and start detection thread
        self.detection_thread = threading.Thread(target=self.realtime_detection_thread)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        try:
            # Wait for detection thread to complete
            self.detection_thread.join()
        except KeyboardInterrupt:
            print("\n用户中断检测")
            self.running = False
            # Wait for detection thread to end
            if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1.0)


def run_radar_realtime(port='/dev/ttyUSB0',
                       baudrate=230400,
                       max_range=800,
                       target_distance=200):
    """
    Run laser radar real-time position detection (without visualization)

    Args:
        port: Serial port
        baudrate: Baud rate
        max_range: Maximum detection range (mm)
        target_distance: Target distance from line (mm)
    """

    # Create radar position correction object
    radar = RadarPositionCorrection(
        port=port,
        baudrate=baudrate,
        max_range=max_range,
        target_distance_from_line=target_distance
    )

    try:
        # Start real-time detection
        radar.start_realtime_detection()
    except Exception as e:
        print(f"Real-time detection error: {e}")


# Single detection function, kept for compatibility
def run_radar(port='/dev/ttyUSB0',
              baudrate=230400,
              max_range=800,
              target_distance=200,
              collection_time=2.0):
    """
    Run laser radar single position detection

    Args:
        port: Serial port
        baudrate: Baud rate
        max_range: Maximum detection range (mm)
        target_distance: Target distance from line (mm)
        collection_time: Data collection time (seconds)

    Returns:
        dict: {
            'success': bool,                    # Whether detection successful
            'horizontal_offset': float,         # Horizontal offset (mm), negative left, positive right
            'vertical_offset': float,           # Vertical offset (mm), negative backward, positive forward
            'angle_offset': float,              # Angle offset (degrees), negative counterclockwise, positive clockwise
            'distance_to_line': float,          # Current distance to line (mm)
            'is_position_ok': bool,             # Whether position is within tolerance
            'line_info': dict or None           # Detected line information
        }
    """

    # Create radar position correction object
    radar = RadarPositionCorrection(
        port=port,
        baudrate=baudrate,
        max_range=max_range,
        target_distance_from_line=target_distance,
        data_collection_time=collection_time
    )

    try:
        # Collect data
        if not radar.collect_data():
            return {
                'success': False,
                'error': 'Failed to collect radar data',
                'horizontal_offset': 0,
                'vertical_offset': 0,
                'angle_offset': 0,
                'distance_to_line': 0,
                'is_position_ok': False,
                'line_info': None
            }

        # Detect rear line
        rear_line, _ = radar.detect_rear_line()

        if rear_line is None:
            return {
                'success': False,
                'error': 'No rear line detected',
                'horizontal_offset': 0,
                'vertical_offset': 0,
                'angle_offset': 0,
                'distance_to_line': 0,
                'is_position_ok': False,
                'line_info': None
            }

        # Calculate position correction parameters
        correction = radar.calculate_position_correction(rear_line)

        if correction is None:
            return {
                'success': False,
                'error': 'Failed to calculate position correction parameters',
                'horizontal_offset': 0,
                'vertical_offset': 0,
                'angle_offset': 0,
                'distance_to_line': 0,
                'is_position_ok': False,
                'line_info': None
            }

        # Output results
        print(f"Position detection results:")
        print(f"  Horizontal offset: {correction['horizontal_offset']:.1f} mm")
        print(f"  Vertical offset: {correction['vertical_offset']:.1f} mm")
        print(f"  Angle offset: {correction['angle_offset']:.1f} degrees")
        print(f"  Distance to line: {correction['distance_to_line']:.1f} mm")
        print(f"  Position status: {'OK' if correction['is_position_ok'] else 'Need adjustment'}")

        # Return results
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
        print(f"Radar detection error: {e}")
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


def main():
    """
    Main function, handles command line arguments and runs corresponding detection mode
    """
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Radar Position Detection Tool')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baudrate', type=int, default=230400, help='Baud rate')
    parser.add_argument('--range', type=int, default=800, help='Maximum detection range (mm)')
    parser.add_argument('--target', type=int, default=200, help='Target distance from line (mm)')
    parser.add_argument('--mode', choices=['single', 'realtime'], default='realtime',
                        help='Detection mode: single or realtime')

    # Parse command line arguments
    args = parser.parse_args()

    if args.mode == 'single':
        # Single detection mode
        result = run_radar(
            port=args.port,
            baudrate=args.baudrate,
            max_range=args.range,
            target_distance=args.target,
            collection_time=2.0
        )

        if result['success']:
            print("\n=== Required Position Correction ===")
            print(f"Horizontal movement: {result['horizontal_offset']:.1f} mm")
            print(f"Vertical movement: {result['vertical_offset']:.1f} mm")
            print(f"Rotation angle: {result['angle_offset']:.1f} degrees")
            print(f"Current distance to line: {result['distance_to_line']:.1f} mm")

            if result['is_position_ok']:
                print("✓ Position within tolerance")
            else:
                print("✗ Position correction needed")
        else:
            print(f"❌ Detection failed: {result.get('error', 'Unknown error')}")
    else:
        # Real-time detection mode
        print("启动实时雷达位置检测...")
        print("按 Ctrl+C 退出")
        run_radar_realtime(
            port=args.port,
            baudrate=args.baudrate,
            max_range=args.range,
            target_distance=args.target
        )


# Program entry point
if __name__ == "__main__":
    main()

