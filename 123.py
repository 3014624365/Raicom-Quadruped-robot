#!/usr/bin/env python3

import serial
import numpy as np
import time
import threading
import queue
from collections import deque
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import cv2


class RadarPositionCorrectionVisual:
    """
    Enhanced Radar Position Correction Class with Comprehensive Visualization
    Includes point cloud, radar polar plot, binary image, Hough detection, and morphological operations
    """

    def __init__(self,
                 # ========== Adjustable Parameters ==========
                 port='/dev/ttyUSB0',
                 baudrate=230400,
                 max_range=800,
                 img_size=500,
                 target_distance_from_line=200,
                 angle_tolerance=5,
                 distance_tolerance=20,
                 min_line_length=10,
                 hough_threshold=40,
                 max_line_gap=10,
                 data_collection_time=0.5,
                 gaussian_blur_size=1,
                 binary_threshold=30,
                 erosion_kernel_size=2,
                 erosion_iterations=1,
                 enable_visualization=True,
                 # ==============================
                 ):

        # Basic parameters (same as original)
        self.port = port
        self.baudrate = baudrate
        self.max_range = max_range
        self.img_size = img_size
        self.resolution = max_range * 2 / img_size
        self.target_distance_from_line = target_distance_from_line
        self.angle_tolerance = angle_tolerance
        self.distance_tolerance = distance_tolerance
        self.min_line_length = min_line_length
        self.hough_threshold = hough_threshold
        self.max_line_gap = max_line_gap
        self.gaussian_blur_size = gaussian_blur_size
        self.binary_threshold = binary_threshold
        self.erosion_kernel_size = erosion_kernel_size
        self.erosion_iterations = erosion_iterations
        self.data_collection_time = data_collection_time

        # Visualization parameters
        self.enable_visualization = enable_visualization

        # Running status
        self.running = False
        self.scan_data = {}
        self.data_queue = queue.Queue(maxsize=200)
        self.point_buffer = deque(maxlen=1000)
        self.last_update_time = 0
        self.latest_correction = None

        # Visualization data storage
        self.vis_data = {
            'point_cloud_img': None,
            'radar_polar_img': None,
            'binary_img': None,
            'morphological_img': None,
            'hough_result_img': None,
            'detected_line': None
        }

        # Initialize visualization if enabled
        if self.enable_visualization:
            self.init_visualization()

    def init_visualization(self):
        """Initialize visualization windows and plots"""
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('Real-time Radar Position Detection System', fontsize=16, fontweight='bold')

        # Configure subplots
        self.ax_point_cloud = self.axes[0, 0]
        self.ax_radar_polar = self.axes[0, 1]
        self.ax_binary = self.axes[0, 2]
        self.ax_morphological = self.axes[1, 0]
        self.ax_hough = self.axes[1, 1]
        self.ax_info = self.axes[1, 2]

        # Set titles
        self.ax_point_cloud.set_title('Point Cloud Visualization', fontsize=12, fontweight='bold')
        self.ax_radar_polar.set_title('Radar Polar Plot', fontsize=12, fontweight='bold')
        self.ax_binary.set_title('Binary Image', fontsize=12, fontweight='bold')
        self.ax_morphological.set_title('Morphological Operations', fontsize=12, fontweight='bold')
        self.ax_hough.set_title('Hough Line Detection', fontsize=12, fontweight='bold')
        self.ax_info.set_title('Detection Information', fontsize=12, fontweight='bold')

        # Configure point cloud plot
        self.ax_point_cloud.set_xlim(-self.max_range, self.max_range)
        self.ax_point_cloud.set_ylim(-self.max_range, self.max_range)
        self.ax_point_cloud.set_xlabel('X Position (mm)')
        self.ax_point_cloud.set_ylabel('Y Position (mm)')
        self.ax_point_cloud.grid(True, alpha=0.3)
        self.ax_point_cloud.set_aspect('equal')

        # Configure radar polar plot
        self.ax_radar_polar.set_xlim(-self.max_range, self.max_range)
        self.ax_radar_polar.set_ylim(-self.max_range, self.max_range)
        self.ax_radar_polar.set_xlabel('X Position (mm)')
        self.ax_radar_polar.set_ylabel('Y Position (mm)')
        self.ax_radar_polar.grid(True, alpha=0.3)
        self.ax_radar_polar.set_aspect('equal')

        # Add radar range circles
        for r in range(100, self.max_range + 1, 100):
            circle = patches.Circle((0, 0), r, fill=False, color='gray', alpha=0.3)
            self.ax_radar_polar.add_patch(circle)

        # Configure binary and morphological plots
        for ax in [self.ax_binary, self.ax_morphological, self.ax_hough]:
            ax.set_xlim(0, self.img_size)
            ax.set_ylim(0, self.img_size)
            ax.set_xlabel('Image X (pixels)')
            ax.set_ylabel('Image Y (pixels)')

        # Configure info plot
        self.ax_info.axis('off')

        # Add target distance indicator to point cloud
        target_line_y = -self.target_distance_from_line
        self.ax_point_cloud.axhline(y=target_line_y, color='red', linestyle='--',
                                    alpha=0.7, label=f'Target Distance ({self.target_distance_from_line}mm)')
        self.ax_point_cloud.legend()

        plt.tight_layout()
        plt.ion()  # Interactive mode

    def find_frame_header(self, ser):
        """Find data frame header 0x54 0x2C"""
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
        """Parse 47-byte data frame"""
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
            print(f"Error parsing frame data: {e}")
            return None

    def calculate_point_angles(self, start_angle, end_angle, num_points=12):
        """Calculate angle for each measurement point"""
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
        """Data reading thread"""
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
                print(f"Error reading data: {e}")
                time.sleep(0.1)

    def process_realtime_data(self):
        """Process real-time data and update point buffer"""
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                angles = self.calculate_point_angles(data['start_angle'], data['end_angle'])
                timestamp = time.time()

                for i, (distance, intensity) in enumerate(data['points']):
                    if 0 < distance <= self.max_range:
                        angle = angles[i]
                        x = distance * np.sin(np.radians(angle))
                        y = distance * np.cos(np.radians(angle))
                        self.point_buffer.append((x, y, timestamp, intensity))
                        self.scan_data[angle] = (distance, intensity)
        except queue.Empty:
            pass

        # Clear expired data
        current_time = time.time()
        while self.point_buffer and current_time - self.point_buffer[0][2] > self.data_collection_time:
            self.point_buffer.popleft()

    def world_to_img_coords(self, x, y):
        """Convert world coordinates to image coordinates"""
        img_center = self.img_size // 2
        img_x = int(img_center + x / self.resolution)
        img_y = int(img_center - y / self.resolution)
        img_x = max(0, min(img_x, self.img_size - 1))
        img_y = max(0, min(img_y, self.img_size - 1))
        return img_x, img_y

    def create_binary_image(self, use_buffer=False, return_stages=False):
        """
        Create binary image with optional intermediate stages

        Args:
            use_buffer: Use point buffer data if True, scan data if False
            return_stages: Return intermediate processing stages if True

        Returns:
            If return_stages is True: (final_binary, original_img, blurred_img, morphological_img)
            If return_stages is False: final_binary
        """
        # Create blank image
        original_img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        if use_buffer:
            # Use point buffer data
            for x, y, _, intensity in self.point_buffer:
                img_x, img_y = self.world_to_img_coords(x, y)
                # Use intensity for point brightness
                brightness = min(255, max(100, int(intensity * 2)))
                cv2.circle(original_img, (img_x, img_y), 1, brightness, -1)
        else:
            # Use scan data
            for angle, (distance, intensity) in self.scan_data.items():
                if distance <= self.max_range:
                    x = distance * np.sin(np.radians(angle))
                    y = distance * np.cos(np.radians(angle))
                    img_x, img_y = self.world_to_img_coords(x, y)
                    brightness = min(255, max(100, int(intensity * 2)))
                    cv2.circle(original_img, (img_x, img_y), 1, brightness, -1)

        # Gaussian blur
        if self.gaussian_blur_size > 0:
            kernel_size = max(1, self.gaussian_blur_size * 2 + 1)  # Ensure odd number
            blurred_img = cv2.GaussianBlur(original_img, (kernel_size, kernel_size), 0)
        else:
            blurred_img = original_img.copy()

        # Binary thresholding
        _, binary_img = cv2.threshold(blurred_img, self.binary_threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations
        if self.erosion_kernel_size > 0 and self.erosion_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (self.erosion_kernel_size, self.erosion_kernel_size))
            morphological_img = cv2.erode(binary_img, kernel, iterations=self.erosion_iterations)
        else:
            morphological_img = binary_img.copy()

        if return_stages:
            return morphological_img, original_img, blurred_img, binary_img
        else:
            return morphological_img

    def detect_rear_line(self, use_buffer=False):
        """Detect rear line and return visualization data"""
        # Get binary image and intermediate stages
        binary_img, original_img, blurred_img, pre_morphological = self.create_binary_image(
            use_buffer=use_buffer, return_stages=True)

        # Store visualization data
        self.vis_data['point_cloud_img'] = original_img
        self.vis_data['binary_img'] = pre_morphological
        self.vis_data['morphological_img'] = binary_img

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            binary_img,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        # Create Hough result visualization
        hough_result_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

        if lines is None or len(lines) == 0:
            self.vis_data['hough_result_img'] = hough_result_img
            self.vis_data['detected_line'] = None
            return None, binary_img

        # Filter rear lines
        rear_lines = []
        center = self.img_size // 2

        # Draw all detected lines in blue
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for all lines

            # Convert to world coordinates
            wx1 = (x1 - center) * self.resolution
            wy1 = (center - y1) * self.resolution
            wx2 = (x2 - center) * self.resolution
            wy2 = (center - y2) * self.resolution

            # Check if rear line
            if wy1 < -50 and wy2 < -50:
                length = math.sqrt((wx2 - wx1) ** 2 + (wy2 - wy1) ** 2)
                angle = math.degrees(math.atan2(wy2 - wy1, wx2 - wx1))

                rear_lines.append({
                    'x1': wx1, 'y1': wy1, 'x2': wx2, 'y2': wy2,
                    'img_x1': x1, 'img_y1': y1, 'img_x2': x2, 'img_y2': y2,
                    'length': length,
                    'angle': angle
                })

        if not rear_lines:
            self.vis_data['hough_result_img'] = hough_result_img
            self.vis_data['detected_line'] = None
            return None, binary_img

        # Select longest rear line
        rear_lines.sort(key=lambda x: x['length'], reverse=True)
        selected_line = rear_lines[0]

        # Draw selected line in green
        cv2.line(hough_result_img,
                 (selected_line['img_x1'], selected_line['img_y1']),
                 (selected_line['img_x2'], selected_line['img_y2']),
                 (0, 255, 0), 3)  # Green for selected line

        # Draw radar center
        cv2.circle(hough_result_img, (center, center), 5, (0, 0, 255), -1)  # Red dot for radar

        self.vis_data['hough_result_img'] = hough_result_img
        self.vis_data['detected_line'] = selected_line

        return selected_line, binary_img

    def calculate_position_correction(self, line):
        """Calculate position correction parameters"""
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
            'is_position_ok': (abs(horizontal_offset) < self.distance_tolerance and
                               abs(vertical_adjustment) < self.distance_tolerance and
                               abs(angle_with_x) < self.angle_tolerance)
        }

    def update_visualization(self):
        """Update all visualization plots"""
        if not self.enable_visualization:
            return

        # Clear all axes
        for ax in [self.ax_point_cloud, self.ax_radar_polar, self.ax_binary,
                   self.ax_morphological, self.ax_hough]:
            ax.clear()

        # Update point cloud visualization
        self.update_point_cloud_plot()

        # Update radar polar plot
        self.update_radar_polar_plot()

        # Update binary image
        self.update_binary_plot()

        # Update morphological operations result
        self.update_morphological_plot()

        # Update Hough detection result
        self.update_hough_plot()

        # Update information display
        self.update_info_display()

        # Refresh the display
        plt.draw()
        plt.pause(0.01)

    def update_point_cloud_plot(self):
        """Update point cloud visualization"""
        self.ax_point_cloud.set_title('Point Cloud Visualization', fontsize=12, fontweight='bold')
        self.ax_point_cloud.set_xlim(-self.max_range, self.max_range)
        self.ax_point_cloud.set_ylim(-self.max_range, self.max_range)
        self.ax_point_cloud.set_xlabel('X Position (mm)')
        self.ax_point_cloud.set_ylabel('Y Position (mm)')
        self.ax_point_cloud.grid(True, alpha=0.3)
        self.ax_point_cloud.set_aspect('equal')

        # Plot current points
        if self.point_buffer:
            points_x = [p[0] for p in self.point_buffer]
            points_y = [p[1] for p in self.point_buffer]
            intensities = [p[3] for p in self.point_buffer]

            scatter = self.ax_point_cloud.scatter(points_x, points_y, c=intensities,
                                                  cmap='viridis', s=10, alpha=0.7)

            # Add colorbar for intensity
            if hasattr(self, '_colorbar_point_cloud'):
                self._colorbar_point_cloud.remove()
            self._colorbar_point_cloud = plt.colorbar(scatter, ax=self.ax_point_cloud,
                                                      label='Intensity', shrink=0.8)

        # Add radar center
        self.ax_point_cloud.plot(0, 0, 'ro', markersize=8, label='Radar Position')

        # Add target distance line
        target_line_y = -self.target_distance_from_line
        self.ax_point_cloud.axhline(y=target_line_y, color='red', linestyle='--',
                                    alpha=0.7, label=f'Target Distance ({self.target_distance_from_line}mm)')

        # Add detected line if available
        if self.vis_data['detected_line']:
            line = self.vis_data['detected_line']
            self.ax_point_cloud.plot([line['x1'], line['x2']], [line['y1'], line['y2']],
                                     'g-', linewidth=3, label='Detected Line')

        self.ax_point_cloud.legend(fontsize=8)

    def update_radar_polar_plot(self):
        """Update radar polar plot"""
        self.ax_radar_polar.set_title('Radar Polar Plot', fontsize=12, fontweight='bold')
        self.ax_radar_polar.set_xlim(-self.max_range, self.max_range)
        self.ax_radar_polar.set_ylim(-self.max_range, self.max_range)
        self.ax_radar_polar.set_xlabel('X Position (mm)')
        self.ax_radar_polar.set_ylabel('Y Position (mm)')
        self.ax_radar_polar.grid(True, alpha=0.3)
        self.ax_radar_polar.set_aspect('equal')

        # Add radar range circles
        for r in range(100, self.max_range + 1, 100):
            circle = patches.Circle((0, 0), r, fill=False, color='gray', alpha=0.3)
            self.ax_radar_polar.add_patch(circle)

        # Add angle lines
        for angle in range(0, 360, 30):
            x = self.max_range * np.sin(np.radians(angle))
            y = self.max_range * np.cos(np.radians(angle))
            self.ax_radar_polar.plot([0, x], [0, y], 'k-', alpha=0.2)

        # Plot scan data points
        if self.scan_data:
            for angle, (distance, intensity) in self.scan_data.items():
                if distance <= self.max_range:
                    x = distance * np.sin(np.radians(angle))
                    y = distance * np.cos(np.radians(angle))
                    color_intensity = intensity / 255.0
                    self.ax_radar_polar.plot(x, y, 'o', color=(1 - color_intensity, color_intensity, 0),
                                             markersize=3, alpha=0.8)

        # Add radar center
        self.ax_radar_polar.plot(0, 0, 'ro', markersize=8, label='Radar')
        self.ax_radar_polar.legend(fontsize=8)

    def update_binary_plot(self):
        """Update binary image plot"""
        self.ax_binary.set_title('Binary Image', fontsize=12, fontweight='bold')

        if self.vis_data['binary_img'] is not None:
            self.ax_binary.imshow(self.vis_data['binary_img'], cmap='gray', origin='upper')

        self.ax_binary.set_xlabel('Image X (pixels)')
        self.ax_binary.set_ylabel('Image Y (pixels)')

    def update_morphological_plot(self):
        """Update morphological operations result"""
        self.ax_morphological.set_title('Morphological Operations', fontsize=12, fontweight='bold')

        if self.vis_data['morphological_img'] is not None:
            self.ax_morphological.imshow(self.vis_data['morphological_img'], cmap='gray', origin='upper')

        self.ax_morphological.set_xlabel('Image X (pixels)')
        self.ax_morphological.set_ylabel('Image Y (pixels)')

    def update_hough_plot(self):
        """Update Hough line detection result"""
        self.ax_hough.set_title('Hough Line Detection', fontsize=12, fontweight='bold')

        if self.vis_data['hough_result_img'] is not None:
            # Convert BGR to RGB for matplotlib
            hough_rgb = cv2.cvtColor(self.vis_data['hough_result_img'], cv2.COLOR_BGR2RGB)
            self.ax_hough.imshow(hough_rgb, origin='upper')

        self.ax_hough.set_xlabel('Image X (pixels)')
        self.ax_hough.set_ylabel('Image Y (pixels)')

    def update_info_display(self):
        """Update information display"""
        self.ax_info.clear()
        self.ax_info.set_title('Detection Information', fontsize=12, fontweight='bold')
        self.ax_info.axis('off')

        info_text = []
        info_text.append("RADAR STATUS:")
        info_text.append(f"• Points in buffer: {len(self.point_buffer)}")
        info_text.append(f"• Scan angles: {len(self.scan_data)}")
        info_text.append("")

        if self.latest_correction:
            info_text.append("POSITION CORRECTION:")
            info_text.append(f"• Horizontal offset: {self.latest_correction['horizontal_offset']:.1f} mm")
            info_text.append(f"• Vertical offset: {self.latest_correction['vertical_offset']:.1f} mm")
            info_text.append(f"• Angle offset: {self.latest_correction['angle_offset']:.1f}°")
            info_text.append(f"• Distance to line: {self.latest_correction['distance_to_line']:.1f} mm")
            info_text.append(f"• Target distance: {self.target_distance_from_line} mm")
            info_text.append("")

            status_color = 'green' if self.latest_correction['is_position_ok'] else 'red'
            status_text = 'POSITION OK' if self.latest_correction['is_position_ok'] else 'ADJUSTMENT NEEDED'
            info_text.append(f"STATUS: {status_text}")
        else:
            info_text.append("LINE DETECTION:")
            info_text.append("• No rear line detected")
            info_text.append("• Check radar positioning")

        # Display text
        text_y = 0.9
        for line in info_text:
            if line.startswith("STATUS:"):
                color = 'green' if 'OK' in line else 'red'
                self.ax_info.text(0.05, text_y, line, transform=self.ax_info.transAxes,
                                  fontsize=10, fontweight='bold', color=color)
            else:
                self.ax_info.text(0.05, text_y, line, transform=self.ax_info.transAxes,
                                  fontsize=10, fontfamily='monospace')
            text_y -= 0.08

    def realtime_detection_thread(self):
        """Real-time detection thread with visualization"""
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.port}")
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            return

        self.running = True
        self.scan_data = {}
        self.point_buffer.clear()

        # Start data reading thread
        reader_thread = threading.Thread(target=self.data_reader_thread, args=(ser,))
        reader_thread.daemon = True
        reader_thread.start()

        print("Starting real-time detection with visualization...")
        print("Close the plot window or press Ctrl+C to stop")

        try:
            while self.running:
                # Process data in queue
                self.process_realtime_data()

                # Update detection and visualization periodically
                current_time = time.time()
                if current_time - self.last_update_time >= 0.5 and len(self.point_buffer) >= 50:
                    # Detect rear line
                    line, binary_img = self.detect_rear_line(use_buffer=True)

                    if line is not None:
                        self.latest_correction = self.calculate_position_correction(line)
                        if self.latest_correction:
                            print(f"Distance: {self.latest_correction['distance_to_line']:.1f}mm | "
                                  f"Target: {self.target_distance_from_line}mm | "
                                  f"H_offset: {self.latest_correction['horizontal_offset']:.1f}mm | "
                                  f"V_offset: {self.latest_correction['vertical_offset']:.1f}mm | "
                                  f"Angle: {self.latest_correction['angle_offset']:.1f}° | "
                                  f"Status: {'OK' if self.latest_correction['is_position_ok'] else 'ADJUST'}")
                    else:
                        self.latest_correction = None
                        print("No rear line detected")

                    # Update visualization
                    if self.enable_visualization:
                        self.update_visualization()

                    self.last_update_time = current_time

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nUser interrupted detection")
        except Exception as e:
            print(f"Real-time detection error: {e}")
        finally:
            finally:
            self.running = False
            ser.close()
            print("Real-time detection stopped")

    def start_realtime_detection(self):
        """Start real-time detection with visualization"""
        # Create and start detection thread
        self.detection_thread = threading.Thread(target=self.realtime_detection_thread)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        try:
            # Keep the plot window open
            if self.enable_visualization:
                plt.show(block=True)
            else:
                # Wait for detection thread to complete
                self.detection_thread.join()
        except KeyboardInterrupt:
            print("\nUser interrupted detection")
            self.running = False
            if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1.0)

    def collect_data(self):
        """Collect radar data for single detection mode"""
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.port}")
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            return False

        self.running = True
        self.scan_data = {}

        reader_thread = threading.Thread(target=self.data_reader_thread, args=(ser,))
        reader_thread.daemon = True
        reader_thread.start()

        start_time = time.time()
        data_count = 0

        print(f"Collecting data for {self.data_collection_time} seconds...")

        while time.time() - start_time < self.data_collection_time:
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get_nowait()
                    angles = self.calculate_point_angles(data['start_angle'], data['end_angle'])

                    for i, (distance, intensity) in enumerate(data['points']):
                        if 0 < distance <= self.max_range:
                            angle = angles[i]
                            self.scan_data[angle] = (distance, intensity)
                            data_count += 1

                time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error collecting data: {e}")
                break

        self.running = False
        ser.close()

        print(f"Data collection complete. Collected {data_count} data points, {len(self.scan_data)} valid angles")
        return len(self.scan_data) > 50

    def run_single_detection_with_visualization(self):
        """Run single detection with comprehensive visualization"""
        print("Starting single detection with visualization...")

        # Collect data
        if not self.collect_data():
            print("Failed to collect radar data")
            return None

        # Initialize visualization if not already done
        if self.enable_visualization and not hasattr(self, 'fig'):
            self.init_visualization()

        # Detect rear line
        rear_line, _ = self.detect_rear_line(use_buffer=False)

        if rear_line is None:
            print("No rear line detected")
            correction = None
        else:
            # Calculate position correction parameters
            correction = self.calculate_position_correction(rear_line)

        # Update visualization
        if self.enable_visualization:
            self.latest_correction = correction
            self.update_visualization()

            # Add title indicating single detection mode
            self.fig.suptitle('Single Detection Mode - Radar Position Correction',
                              fontsize=16, fontweight='bold')

            plt.show(block=True)

        # Print results
        if correction:
            print(f"\nPosition detection results:")
            print(f"  Horizontal offset: {correction['horizontal_offset']:.1f} mm")
            print(f"  Vertical offset: {correction['vertical_offset']:.1f} mm")
            print(f"  Angle offset: {correction['angle_offset']:.1f} degrees")
            print(f"  Distance to line: {correction['distance_to_line']:.1f} mm")
            print(f"  Position status: {'OK' if correction['is_position_ok'] else 'Need adjustment'}")

        return correction

    def run_radar_realtime_visual(port='/dev/ttyUSB0',
                                  baudrate=230400,
                                  max_range=800,
                                  target_distance=200):
        """
        Run laser radar real-time position detection with comprehensive visualization

        Args:
            port: Serial port
            baudrate: Baud rate
            max_range: Maximum detection range (mm)
            target_distance: Target distance from line (mm)
        """
        radar = RadarPositionCorrectionVisual(
            port=port,
            baudrate=baudrate,
            max_range=max_range,
            target_distance_from_line=target_distance,
            enable_visualization=True
        )

        try:
            radar.start_realtime_detection()
        except Exception as e:
            print(f"Real-time detection error: {e}")

    def run_radar_single_visual(port='/dev/ttyUSB0',
                                baudrate=230400,
                                max_range=800,
                                target_distance=200,
                                collection_time=2.0):
        """
        Run laser radar single position detection with visualization

        Args:
            port: Serial port
            baudrate: Baud rate
            max_range: Maximum detection range (mm)
            target_distance: Target distance from line (mm)
            collection_time: Data collection time (seconds)

        Returns:
            dict: Detection results
        """
        radar = RadarPositionCorrectionVisual(
            port=port,
            baudrate=baudrate,
            max_range=max_range,
            target_distance_from_line=target_distance,
            data_collection_time=collection_time,
            enable_visualization=True
        )

        try:
            correction = radar.run_single_detection_with_visualization()

            if correction is None:
                return {
                    'success': False,
                    'error': 'No rear line detected or data collection failed',
                    'horizontal_offset': 0,
                    'vertical_offset': 0,
                    'angle_offset': 0,
                    'distance_to_line': 0,
                    'is_position_ok': False,
                    'line_info': None
                }

            return {
                'success': True,
                'horizontal_offset': correction['horizontal_offset'],
                'vertical_offset': correction['vertical_offset'],
                'angle_offset': correction['angle_offset'],
                'distance_to_line': correction['distance_to_line'],
                'is_position_ok': correction['is_position_ok'],
                'line_info': radar.vis_data['detected_line']
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

        # Keep original functions for backward compatibility

    def run_radar_realtime(port='/dev/ttyUSB0', baudrate=230400, max_range=800, target_distance=200):
        """Original non-visual real-time detection function"""
        from RadarPositionCorrection import RadarPositionCorrection

        radar = RadarPositionCorrection(
            port=port,
            baudrate=baudrate,
            max_range=max_range,
            target_distance_from_line=target_distance
        )

        try:
            radar.start_realtime_detection()
        except Exception as e:
            print(f"Real-time detection error: {e}")

    def run_radar(port='/dev/ttyUSB0', baudrate=230400, max_range=800, target_distance=200, collection_time=2.0):
        """Original single detection function"""
        # You would need to import your original class here
        from RadarPositionCorrection import RadarPositionCorrection

        radar = RadarPositionCorrection(
            port=port,
            baudrate=baudrate,
            max_range=max_range,
            target_distance_from_line=target_distance,
            data_collection_time=collection_time
        )

        try:
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

            print(f"Position detection results:")
            print(f"  Horizontal offset: {correction['horizontal_offset']:.1f} mm")
            print(f"  Vertical offset: {correction['vertical_offset']:.1f} mm")
            print(f"  Angle offset: {correction['angle_offset']:.1f} degrees")
            print(f"  Distance to line: {correction['distance_to_line']:.1f} mm")
            print(f"  Position status: {'OK' if correction['is_position_ok'] else 'Need adjustment'}")

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
        """Enhanced main function with visualization options"""
        parser = argparse.ArgumentParser(description='Radar Position Detection Tool with Visualization')
        parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port')
        parser.add_argument('--baudrate', type=int, default=230400, help='Baud rate')
        parser.add_argument('--range', type=int, default=800, help='Maximum detection range (mm)')
        parser.add_argument('--target', type=int, default=200, help='Target distance from line (mm)')
        parser.add_argument('--mode', choices=['single', 'realtime', 'single-visual', 'realtime-visual'],
                            default='realtime-visual',
                            help='Detection mode: single, realtime, single-visual, or realtime-visual')
        parser.add_argument('--collection-time', type=float, default=2.0,
                            help='Data collection time for single mode (seconds)')

        args = parser.parse_args()

        print(f"Starting radar detection in {args.mode} mode...")
        print("Available modes:")
        print("  single: Single detection without visualization")
        print("  realtime: Real-time detection without visualization")
        print("  single-visual: Single detection with comprehensive visualization")
        print("  realtime-visual: Real-time detection with comprehensive visualization")
        print()

        if args.mode == 'single':
            result = run_radar(
                port=args.port,
                baudrate=args.baudrate,
                max_range=args.range,
                target_distance=args.target,
                collection_time=args.collection_time
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

        elif args.mode == 'realtime':
            print("Starting real-time radar position detection...")
            print("Press Ctrl+C to exit")
            run_radar_realtime(
                port=args.port,
                baudrate=args.baudrate,
                max_range=args.range,
                target_distance=args.target
            )

        elif args.mode == 'single-visual':
            print("Starting single detection with comprehensive visualization...")
            print("Close the plot window to exit")
            result = run_radar_single_visual(
                port=args.port,
                baudrate=args.baudrate,
                max_range=args.range,
                target_distance=args.target,
                collection_time=args.collection_time
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

        elif args.mode == 'realtime-visual':
            print("Starting real-time radar position detection with visualization...")
            print("Close the plot window or press Ctrl+C to exit")
            run_radar_realtime_visual(
                port=args.port,
                baudrate=args.baudrate,
                max_range=args.range,
                target_distance=args.target
            )

        # Program entry point

    if __name__ == "__main__":
        main()

