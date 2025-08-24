#!/usr/bin/env python3

import serial  # For serial communication
import numpy as np  # For numerical computation
import cv2  # For image processing
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
                 display_scale=1.5,  # Display scale ratio
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
        self.display_scale = display_scale  # Display scale ratio

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
        self.visualization_img = None  # Visualization image

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

    def draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=10):
        """
        Draw dashed line (replacement for OpenCV's LINE_DASHED)

        Args:
            img: Image
            pt1: Start point (x1, y1)
            pt2: End point (x2, y2)
            color: Color (B, G, R)
            thickness: Line width
            dash_length: Dash segment length
            gap_length: Gap length
        """
        # Calculate total line length
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

        # Calculate number of dash segments needed
        dashes = int(dist / (dash_length + gap_length))

        # If distance too short, draw solid line directly
        if dashes == 0:
            cv2.line(img, pt1, pt2, color, thickness)
            return

        # Draw each dash segment
        for i in range(dashes):
            # Calculate start and end points for each dash segment
            start_factor = i * (dash_length + gap_length) / dist
            end_factor = min(1, (i * (dash_length + gap_length) + dash_length) / dist)

            # Calculate start point coordinates
            start_pt = (int(pt1[0] + (pt2[0] - pt1[0]) * start_factor),
                        int(pt1[1] + (pt2[1] - pt1[1]) * start_factor))

            # Calculate end point coordinates
            end_pt = (int(pt1[0] + (pt2[0] - pt1[0]) * end_factor),
                      int(pt1[1] + (pt2[1] - pt1[1]) * end_factor))

            # Draw line segment
            cv2.line(img, start_pt, end_pt, color, thickness)

    def create_visualization(self, binary_img, line, correction):
        """
        Create visualization image showing detected line and position information

        Args:
            binary_img: Binary image
            line: Detected line information
            correction: Position correction parameters

        Returns:
            ndarray: Visualization image
        """
        # Create color image
        vis_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        center = self.img_size // 2  # Image center (radar position)

        # Draw coordinate axes
        cv2.line(vis_img, (0, center), (self.img_size, center), (0, 255, 0), 1)  # X-axis (green)
        cv2.line(vis_img, (center, 0), (center, self.img_size), (0, 255, 0), 1)  # Y-axis (green)

        # Draw radar position
        cv2.circle(vis_img, (center, center), 5, (0, 0, 255), -1)  # Radar position (red circle)

        # If line detected, draw line and related information
        if line is not None:
            # Draw detected rear line (blue)
            cv2.line(vis_img, (line['img_x1'], line['img_y1']), (line['img_x2'], line['img_y2']), (255, 0, 0), 2)

            # Calculate target line position (parallel to current line but at target_distance)
            if correction is not None:
                # Calculate line midpoint
                mid_x = (line['img_x1'] + line['img_x2']) // 2
                mid_y = (line['img_y1'] + line['img_y2']) // 2

                # Calculate unit normal vector of line (unit vector perpendicular to line)
                dx = line['img_x2'] - line['img_x1']  # X direction difference
                dy = line['img_y2'] - line['img_y1']  # Y direction difference
                length = math.sqrt(dx * dx + dy * dy)  # Length
                nx = -dy / length  # Normal vector x component (normalized)
                ny = dx / length  # Normal vector y component (normalized)

                # Calculate two endpoints of target line (offset along normal vector by target_distance)
                target_dist_px = self.target_distance_from_line / self.resolution  # Convert to pixels
                target_x1 = int(line['img_x1'] + nx * target_dist_px)
                target_y1 = int(line['img_y1'] + ny * target_dist_px)
                target_x2 = int(line['img_x2'] + nx * target_dist_px)
                target_y2 = int(line['img_y2'] + ny * target_dist_px)

                # Draw target line (yellow dashed line)
                self.draw_dashed_line(vis_img, (target_x1, target_y1), (target_x2, target_y2), (0, 255, 255), 1)

                # Draw current distance line (perpendicular from radar position to line, yellow)
                end_x = int(center + nx * (correction['distance_to_line'] / self.resolution))
                end_y = int(center + ny * (correction['distance_to_line'] / self.resolution))
                cv2.line(vis_img, (center, center), (end_x, end_y), (255, 255, 0), 2)

        # Add information text
        info_lines = []
        if correction is not None:
            # Get various offsets
            h_offset = correction['horizontal_offset']  # Horizontal offset
            v_offset = correction['vertical_offset']  # Vertical offset
            a_offset = correction['angle_offset']  # Angle offset

            # Position status
            status = "✓ Position OK" if correction['is_position_ok'] else "✗ Need Adjust"

            # Build display text
            info_lines = [
                f"Horizontal: {h_offset:.1f} mm {'←' if h_offset > 0 else '→' if h_offset < 0 else ''}",
                f"Vertical: {v_offset:.1f} mm {'↑' if v_offset > 0 else '↓' if v_offset < 0 else ''}",
                f"Angle: {a_offset:.1f}° {'↺' if a_offset > 0 else '↻' if a_offset < 0 else ''}",
                f"Distance: {correction['distance_to_line']:.1f} mm",
                f"Target: {self.target_distance_from_line} mm",
                status
            ]
        else:
            info_lines = ["No rear line detected"]

        # Add text to image
        for i, line_text in enumerate(info_lines):
            y = 30 + i * 30  # Text y coordinate
            # Add text shadow (white)
            cv2.putText(vis_img, line_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Add text (red)
            cv2.putText(vis_img, line_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        return vis_img

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
                        # Create visualization image
                        self.visualization_img = self.create_visualization(binary_img, line, self.latest_correction)
                    else:
                        self.latest_correction = None
                        self.visualization_img = self.create_visualization(binary_img, None, None)

                    # Update last update time
                    self.last_update_time = current_time

                time.sleep(0.01)  # Brief sleep to avoid high CPU usage

        except KeyboardInterrupt:
            print("User interrupted detection")
        except Exception as e:
            print(f"Real-time detection error: {e}")
        finally:
            # Clean up resources
            self.running = False
            ser.close()
            print("Real-time detection stopped")

    def start_realtime_detection(self):
        """
        Start real-time detection including detection thread and display window
        """
        # Create and start detection thread
        self.detection_thread = threading.Thread(target=self.realtime_detection_thread)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        # Create display window
        cv2.namedWindow("Radar Position Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Radar Position Detection", int(self.img_size * self.display_scale),
                         int(self.img_size * self.display_scale))
        try:
            # Display image loop
            while self.running:
                if self.visualization_img is not None:
                    cv2.imshow("Radar Position Detection", self.visualization_img)

                # Check key press
                key = cv2.waitKey(10)
                if key == 27 or key == ord('q'):  # ESC or q key to exit
                    break

        except KeyboardInterrupt:
            print("User interrupted display")
        finally:
            # Clean up resources
            self.running = False
            cv2.destroyAllWindows()
            # Wait for detection thread to end
            if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1.0)


def run_radar_realtime(port='/dev/ttyUSB0',
                       baudrate=230400,
                       max_range=800,
                       target_distance=200):
    """
    Run laser radar real-time position detection

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
        print("Starting real-time radar position detection...")
        print("Press ESC or q to exit")
        run_radar_realtime(
            port=args.port,
            baudrate=args.baudrate,
            max_range=args.range,
            target_distance=args.target
        )


# Program entry point
if __name__ == "__main__":
    main()
