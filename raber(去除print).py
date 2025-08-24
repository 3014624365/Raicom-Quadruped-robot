import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time

import dog


class RGBMaskDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("实时RGB色域掩膜检测")
        self.root.geometry("1200x800")

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # RGB阈值初始值
        self.r_min = tk.IntVar(value=0)
        self.r_max = tk.IntVar(value=255)
        self.g_min = tk.IntVar(value=0)
        self.g_max = tk.IntVar(value=255)
        self.b_min = tk.IntVar(value=0)
        self.b_max = tk.IntVar(value=255)

        # 运行标志
        self.running = False

        self.setup_ui()

    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="RGB阈值控制", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # 右侧视频显示区域
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 创建RGB滑块
        self.create_rgb_sliders(control_frame)

        # 控制按钮
        self.create_control_buttons(control_frame)

        # 预设颜色按钮
        self.create_preset_buttons(control_frame)

        # 视频显示标签
        self.create_video_labels(video_frame)

    def create_rgb_sliders(self, parent):
        slider_frame = ttk.Frame(parent)
        slider_frame.pack(fill=tk.X, padx=10, pady=10)

        # R通道滑块
        ttk.Label(slider_frame, text="红色通道 (R)", font=("Arial", 12, "bold")).pack(pady=(0, 5))

        r_frame = ttk.Frame(slider_frame)
        r_frame.pack(fill=tk.X, pady=5)
        ttk.Label(r_frame, text="最小值:").pack()
        self.r_min_scale = ttk.Scale(r_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                     variable=self.r_min, length=200)
        self.r_min_scale.pack(fill=tk.X)
        self.r_min_label = ttk.Label(r_frame, text="0")
        self.r_min_label.pack()

        ttk.Label(r_frame, text="最大值:").pack()
        self.r_max_scale = ttk.Scale(r_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                     variable=self.r_max, length=200)
        self.r_max_scale.pack(fill=tk.X)
        self.r_max_label = ttk.Label(r_frame, text="255")
        self.r_max_label.pack()

        # G通道滑块
        ttk.Label(slider_frame, text="绿色通道 (G)", font=("Arial", 12, "bold")).pack(pady=(20, 5))

        g_frame = ttk.Frame(slider_frame)
        g_frame.pack(fill=tk.X, pady=5)
        ttk.Label(g_frame, text="最小值:").pack()
        self.g_min_scale = ttk.Scale(g_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                     variable=self.g_min, length=200)
        self.g_min_scale.pack(fill=tk.X)
        self.g_min_label = ttk.Label(g_frame, text="0")
        self.g_min_label.pack()

        ttk.Label(g_frame, text="最大值:").pack()
        self.g_max_scale = ttk.Scale(g_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                     variable=self.g_max, length=200)
        self.g_max_scale.pack(fill=tk.X)
        self.g_max_label = ttk.Label(g_frame, text="255")
        self.g_max_label.pack()

        # B通道滑块
        ttk.Label(slider_frame, text="蓝色通道 (B)", font=("Arial", 12, "bold")).pack(pady=(20, 5))

        b_frame = ttk.Frame(slider_frame)
        b_frame.pack(fill=tk.X, pady=5)
        ttk.Label(b_frame, text="最小值:").pack()
        self.b_min_scale = ttk.Scale(b_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                     variable=self.b_min, length=200)
        self.b_min_scale.pack(fill=tk.X)
        self.b_min_label = ttk.Label(b_frame, text="0")
        self.b_min_label.pack()

        ttk.Label(b_frame, text="最大值:").pack()
        self.b_max_scale = ttk.Scale(b_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                     variable=self.b_max, length=200)
        self.b_max_scale.pack(fill=tk.X)
        self.b_max_label = ttk.Label(b_frame, text="255")
        self.b_max_label.pack()

        # 绑定滑块变化事件
        for scale, label in [(self.r_min_scale, self.r_min_label), (self.r_max_scale, self.r_max_label),
                             (self.g_min_scale, self.g_min_label), (self.g_max_scale, self.g_max_label),
                             (self.b_min_scale, self.b_min_label), (self.b_max_scale, self.b_max_label)]:
            scale.configure(command=lambda val, l=label: l.configure(text=str(int(float(val)))))

    def create_control_buttons(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=10, pady=20)

        self.start_button = ttk.Button(button_frame, text="开始检测", command=self.start_detection)
        self.start_button.pack(fill=tk.X, pady=2)

        self.stop_button = ttk.Button(button_frame, text="停止检测", command=self.stop_detection)
        self.stop_button.pack(fill=tk.X, pady=2)

        ttk.Button(button_frame, text="重置参数", command=self.reset_values).pack(fill=tk.X, pady=2)

    def create_preset_buttons(self, parent):
        preset_frame = ttk.LabelFrame(parent, text="预设颜色")
        preset_frame.pack(fill=tk.X, padx=10, pady=10)

        presets = [
            ("红色", (0, 100, 100, 255, 255, 255)),
            ("绿色", (0, 255, 0, 255, 100, 255)),
            ("蓝色", (100, 255, 0, 255, 0, 100)),
            ("黄色", (0, 255, 0, 255, 0, 100)),
            ("青色", (100, 255, 100, 255, 0, 255)),
            ("品红", (100, 255, 0, 100, 100, 255))
        ]

        for i, (name, values) in enumerate(presets):
            if i % 2 == 0:
                row_frame = ttk.Frame(preset_frame)
                row_frame.pack(fill=tk.X, pady=2)

            ttk.Button(row_frame, text=name,
                       command=lambda v=values: self.set_preset_values(v)).pack(side=tk.LEFT, expand=True, fill=tk.X,
                                                                                padx=1)

    def create_video_labels(self, parent):
        # 原始视频标签
        ttk.Label(parent, text="原始视频", font=("Arial", 14, "bold")).pack()
        self.original_label = ttk.Label(parent)
        self.original_label.pack(pady=10)

        # 掩膜视频标签
        ttk.Label(parent, text="掩膜结果", font=("Arial", 14, "bold")).pack()
        self.mask_label = ttk.Label(parent)
        self.mask_label.pack(pady=10)

    def set_preset_values(self, values):
        self.r_min.set(values[0])
        self.r_max.set(values[1])
        self.g_min.set(values[2])
        self.g_max.set(values[3])
        self.b_min.set(values[4])
        self.b_max.set(values[5])

        # 更新标签显示
        self.r_min_label.configure(text=str(values[0]))
        self.r_max_label.configure(text=str(values[1]))
        self.g_min_label.configure(text=str(values[2]))
        self.g_max_label.configure(text=str(values[3]))
        self.b_min_label.configure(text=str(values[4]))
        self.b_max_label.configure(text=str(values[5]))

    def reset_values(self):
        self.r_min.set(0)
        self.r_max.set(255)
        self.g_min.set(0)
        self.g_max.set(255)
        self.b_min.set(0)
        self.b_max.set(255)

        # 更新标签显示
        for label in [self.r_min_label, self.g_min_label, self.b_min_label]:
            label.configure(text="0")
        for label in [self.r_max_label, self.g_max_label, self.b_max_label]:
            label.configure(text="255")

    def start_detection(self):
        if not self.running:
            self.running = True
            self.start_button.configure(state='disabled')
            self.stop_button.configure(state='normal')
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()

    def stop_detection(self):
        self.running = False
        self.start_button.configure(state='normal')
        self.stop_button.configure(state='disabled')

    def detection_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 创建RGB掩膜
            lower_bound = np.array([self.b_min.get(), self.g_min.get(), self.r_min.get()])
            upper_bound = np.array([self.b_max.get(), self.g_max.get(), self.r_max.get()])

            # 创建掩膜
            mask = cv2.inRange(frame, lower_bound, upper_bound)

            # 应用掩膜到原图像
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # 转换为RGB格式用于显示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masked_frame_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)

            # 调整图像大小
            frame_rgb = cv2.resize(frame_rgb, (400, 300))
            masked_frame_rgb = cv2.resize(masked_frame_rgb, (400, 300))

            # 转换为PIL图像
            img_original = Image.fromarray(frame_rgb)
            img_mask = Image.fromarray(masked_frame_rgb)

            # 转换为PhotoImage
            photo_original = ImageTk.PhotoImage(img_original)
            photo_mask = ImageTk.PhotoImage(img_mask)

            # 更新显示
            self.original_label.configure(image=photo_original)
            self.original_label.image = photo_original

            self.mask_label.configure(image=photo_mask)
            self.mask_label.image = photo_mask

            time.sleep(0.03)  # 控制帧率约30fps

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


def main():
    root = tk.Tk()
    app = RGBMaskDetector(root)

    def on_closing():
        app.running = False
        if hasattr(app, 'cap'):
            app.cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
    dog.adjust_y(-22.3)
    time.sleep(0.3)

    dog.adjust_y((22,3)
    time.sleep(0.3))