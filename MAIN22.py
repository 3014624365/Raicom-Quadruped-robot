import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from matplotlib import rcParams

# 设置全局参数，提高科研风格
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.linewidth'] = 1.2
rcParams['axes.grid'] = True
rcParams['grid.linestyle'] = '--'
rcParams['grid.alpha'] = 0.3
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 5
rcParams['xtick.minor.size'] = 3
rcParams['ytick.minor.size'] = 3

# 修正后的数据：3秒间隔
time = np.arange(0, 81, 3)  # 0到80秒，每3秒一个点
pressure = np.array(
    [0.029, 0.034, 0.036, 0.037, 0.039, 0.040, 0.041, 0.042, 0.043, 0.045, 0.046, 0.047, 0.049, 0.050, 0.051, 0.053,
     0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.060, 0.061, 0.062, 0.063, 0.065])

print(f"时间点数: {len(time)}")
print(f"压强数据点数: {len(pressure)}")
print(f"时间序列: {time}")

# 直接计算dp/dt，不进行平滑处理
dp_dt = np.zeros_like(time, dtype=float)
dp_dt[1:] = np.diff(pressure) / np.diff(time)  # 每段的斜率
dp_dt[0] = dp_dt[1]  # 第一个点使用第二段的值

# 假设V = 1（你可以替换为实际体积值）
V = 1.0
Q_0 = V * dp_dt

# 输出每一段的Q₀值
print("\n每一段的Q₀值:")
for i in range(len(time)):
    if i == 0:
        print(f"t = {time[i]:2.0f}s: Q₀ = {Q_0[i]*1000:6.3f} ×10⁻³ Pa·m³/s (使用第一段斜率)")
    else:
        print(f"t = {time[i]:2.0f}s: Q₀ = {Q_0[i]*1000:6.3f} ×10⁻³ Pa·m³/s (第{i}段: Δp = {pressure[i]-pressure[i-1]:.3f} Pa)")

# Setup Chinese font
try:
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
    chinese_font = None

    for font in chinese_fonts:
        try:
            chinese_font = FontProperties(font=font)
            break
        except:
            continue

    if chinese_font is None:
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            '/System/Library/Fonts/PingFang.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
        ]

        for path in font_paths:
            if fm.os.path.exists(path):
                chinese_font = FontProperties(fname=path)
                break

    if chinese_font is None:
        chinese_font = FontProperties()

except:
    chinese_font = FontProperties()

# 创建高品质科研风格图表
fig, ax = plt.subplots(figsize=(12, 7), dpi=120, facecolor='white')

# 绘制主数据线（原始数据，不平滑）
main_color = '#1565C0'
plt.plot(time, Q_0 * 1000, '-', color=main_color, lw=2, alpha=0.9, label='出气率$Q_0$（未平滑）')

# 添加数据点，使用不同的标记样式
plt.plot(time, Q_0 * 1000, 's', color=main_color, markersize=7,
         markerfacecolor='white', markeredgewidth=1.8, markeredgecolor=main_color)

# 设置坐标轴标签和标题
plt.xlabel('时间 (s)', fontproperties=chinese_font, fontsize=14)
plt.ylabel('出气率 $Q_0$ (×10$^{-3}$ Pa·m$^3$/s)', fontproperties=chinese_font, fontsize=14)
plt.title('时间与出气率$Q_0$的关系图（3秒间隔，未平滑）', fontproperties=chinese_font, fontsize=16, pad=15)

# 自定义网格
plt.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.8)

# 自定义脊线
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#333333')

# 设置坐标轴范围，留出适当边距
y_range = max(Q_0 * 1000) - min(Q_0 * 1000)
y_min = min(Q_0 * 1000) - y_range * 0.1
y_max = max(Q_0 * 1000) + y_range * 0.1
plt.ylim(y_min, y_max)

# 设置x轴范围
plt.xlim(-2, max(time) + 2)

# 添加次要刻度
ax.minorticks_on()
ax.tick_params(which='minor', length=3, width=1, direction='in')
ax.tick_params(which='major', length=5, width=1.2, direction='in')

# 设置x轴主要刻度
plt.xticks(np.arange(0, max(time)+1, 9))  # 每9秒显示一个主要刻度

# 添加信息文本框
info_text = '$Q_0 = V\\frac{dp}{dt}$\n时间间隔: 3秒\n原始数据，未进行平滑处理'
plt.figtext(0.75, 0.18, info_text, fontproperties=chinese_font, fontsize=12,
            bbox=dict(facecolor='white', edgecolor='#CCCCCC', alpha=0.9, boxstyle='round,pad=0.6'))

# 添加统计信息
stats_text = f'数据点数: {len(time)}\n最大值: {max(Q_0*1000):.3f}\n最小值: {min(Q_0*1000):.3f}'
plt.figtext(0.02, 0.18, stats_text, fontsize=10,
            bbox=dict(facecolor='#f8f8f8', edgecolor='#CCCCCC', alpha=0.8, boxstyle='round,pad=0.4'))

# 添加图例
plt.legend(loc='upper left', prop=chinese_font, frameon=True, framealpha=0.9, edgecolor='#CCCCCC')

plt.tight_layout()
plt.show()

# 输出一些统计信息
print(f"\nQ₀统计信息:")
print(f"平均值: {np.mean(Q_0*1000):.6f} ×10⁻³ Pa·m³/s")
print(f"最大值: {np.max(Q_0*1000):.6f} ×10⁻³ Pa·m³/s")
print(f"最小值: {np.min(Q_0*1000):.6f} ×10⁻³ Pa·m³/s")
print(f"标准差: {np.std(Q_0*1000):.6f} ×10⁻³ Pa·m³/s")
