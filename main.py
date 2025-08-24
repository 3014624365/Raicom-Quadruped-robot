import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# 查找系统中可用的中文字体
# 尝试多种常见中文字体，提高成功率
try:
    # 方法1：直接指定中文字体文件路径
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # SimHei黑体
    chinese_font = FontProperties(fname=font_path)

    # 如果上面的路径不存在，可以尝试其他常见字体
    if not fm.os.path.exists(font_path):
        # 尝试微软雅黑
        font_path = 'C:/Windows/Fonts/msyh.ttc'
        chinese_font = FontProperties(fname=font_path)

except:
    # 方法2：使用系统中已有的字体名称
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
    chinese_font = None

    for font in chinese_fonts:
        if font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
            chinese_font = FontProperties(font)
            break

    # 如果还是找不到，使用matplotlib的默认fallback机制
    if chinese_font is None:
        chinese_font = FontProperties()

# 数据
x = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
y = [10000, 1600, 230, 59, 690, 17, 61, 32, 20, 14, 10]

# 设置科研风格
plt.style.use('seaborn-v0_8-whitegrid')  # 使用科研常用的网格风格
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 绘制散点和连线
ax.plot(x, y, 'o-', linewidth=2, markersize=8,
        color='#2E86C1', markerfacecolor='#5DADE2',
        markeredgecolor='#1B4F72', markeredgewidth=1.5,
        label='压强')

# 设置轴标签和标题 - 使用字体属性
ax.set_xlabel('时间 (s)', fontsize=14, fontweight='bold', fontproperties=chinese_font)
ax.set_ylabel('压强 (Pa)', fontsize=14, fontweight='bold', fontproperties=chinese_font)
ax.set_title('压强随时间变化曲线', fontsize=16, fontweight='bold', pad=20, fontproperties=chinese_font)

# 设置刻度
ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
ax.tick_params(axis='both', which='minor', labelsize=10, width=1)

# 添加网格
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# 设置y轴为对数刻度（因为数据跨度很大）
ax.set_yscale('log')

# 美化边框
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_edgecolor('black')

# 添加图例 - 使用字体属性
legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, prop=chinese_font)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 如果需要保存高质量图片
# plt.savefig('pressure_time_plot.png', dpi=300, bbox_inches='tight')
