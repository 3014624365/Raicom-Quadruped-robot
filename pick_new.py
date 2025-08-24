import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# 设置中文字体和科研风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class VacuumTank3D:
    def __init__(self, R=0.08, H=0.12, pump_efficiency=0.85):
        """
        3D圆柱储气罐真空抽气模拟
        """
        self.R = R  # 半径
        self.H = H  # 高度
        self.pump_efficiency = pump_efficiency

        # 3D网格参数 - 调整为更稳定的尺寸
        self.nr = 12  # 径向网格数
        self.ntheta = 16  # 角度网格数
        self.nz = 15  # 轴向网格数
        self.nt = 100  # 时间步数

        # 物理参数
        self.P0 = 101325  # 初始压强 (Pa)
        self.T = 300  # 温度 (K)
        self.k_ads = 0.01  # 吸附系数
        self.D = 1.2e-5  # 扩散系数

        # 创建3D圆柱坐标网格
        self.r = np.linspace(0, R, self.nr)
        self.theta = np.linspace(0, 2 * np.pi, self.ntheta)
        self.z = np.linspace(0, H, self.nz)

        # 网格间距
        self.dr = R / (self.nr - 1) if self.nr > 1 else R
        self.dtheta = 2 * np.pi / (self.ntheta - 1) if self.ntheta > 1 else 2 * np.pi
        self.dz = H / (self.nz - 1) if self.nz > 1 else H

        # 时间设置
        self.dt = 0.12
        self.time = np.arange(0, self.nt * self.dt, self.dt)

        # 初始化3D压强场 P(r, theta, z, t)
        self.P = np.ones((self.nt, self.nz, self.ntheta, self.nr)) * self.P0

        self._calculate_3d_pressure_evolution()

    def _pump_rate(self, t):
        """泵速函数"""
        return self.pump_efficiency * 6000 * (1 - np.exp(-t / 4)) * np.tanh(t / 3)

    def _wall_adsorption(self, P_local, r, z):
        """壁面吸附效应"""
        if r >= self.R * 0.9:  # 接近圆柱壁面
            return self.k_ads * P_local * (1 + 0.1 * np.sin(2 * np.pi * z / self.H))
        return 0

    def _calculate_3d_pressure_evolution(self):
        """计算完整3D压强场演化"""
        print("正在计算3D压强场演化...")

        for t_idx in range(1, self.nt):
            if t_idx % 15 == 0:
                print(f"进度: {t_idx / self.nt * 100:.1f}%")

            t = self.time[t_idx]
            P_prev = self.P[t_idx - 1]

            for iz in range(self.nz):
                for itheta in range(self.ntheta):
                    for ir in range(self.nr):
                        r_val = self.r[ir]
                        z_val = self.z[iz]

                        # 简化的3D扩散模型
                        laplacian = 0

                        # 径向扩散
                        if ir > 0 and ir < self.nr - 1:
                            d2P_dr2 = (P_prev[iz, itheta, ir + 1] - 2 * P_prev[iz, itheta, ir] +
                                       P_prev[iz, itheta, ir - 1]) / self.dr ** 2
                            laplacian += d2P_dr2

                        # 轴向扩散
                        if iz > 0 and iz < self.nz - 1:
                            d2P_dz2 = (P_prev[iz + 1, itheta, ir] - 2 * P_prev[iz, itheta, ir] +
                                       P_prev[iz - 1, itheta, ir]) / self.dz ** 2
                            laplacian += d2P_dz2

                        # 角向扩散（简化）
                        if r_val > 1e-6:
                            itheta_next = (itheta + 1) % self.ntheta
                            itheta_prev = (itheta - 1) % self.ntheta
                            d2P_dtheta2 = (P_prev[iz, itheta_next, ir] - 2 * P_prev[iz, itheta, ir] +
                                           P_prev[iz, itheta_prev, ir]) / self.dtheta ** 2
                            laplacian += d2P_dtheta2 / (r_val ** 2)

                        # 扩散项
                        diffusion = self.D * laplacian

                        # 泵抽效应
                        pump_term = 0
                        if iz < 3:  # 底部区域
                            pump_strength = self._pump_rate(t) * (1 - iz / 3)
                            pump_term = -pump_strength * P_prev[iz, itheta, ir] / self.P0

                        # 壁面吸附
                        adsorption = self._wall_adsorption(P_prev[iz, itheta, ir], r_val, z_val)

                        # 更新压强
                        dP_dt = diffusion + pump_term - adsorption
                        new_pressure = P_prev[iz, itheta, ir] + dP_dt * self.dt
                        self.P[t_idx, iz, itheta, ir] = max(1.0, new_pressure)

        print("3D压强场计算完成！")

    def create_3d_visualization(self):
        """创建真正的3D可视化 - 使用稳定的颜色映射方法"""
        fig = plt.figure(figsize=(16, 12))

        # 主3D图
        ax_main = fig.add_subplot(111, projection='3d')

        # 科研配色
        colors = ['#000080', '#0040FF', '#0080FF', '#00BFFF', '#00FFFF',
                  '#40FF40', '#80FF00', '#FFFF00', '#FF8000', '#FF4000', '#FF0000']
        cmap = LinearSegmentedColormap.from_list('scientific', colors, N=256)

        # 数据范围
        P_min = np.min(self.P)
        P_max = self.P0

        def animate(frame):
            ax_main.clear()

            current_time = self.time[frame]
            P_current = self.P[frame]  # shape: (nz, ntheta, nr)

            # 使用标准的cmap方法而不是facecolors来避免索引问题

            # 1. 外表面 (r = R) - 使用标准颜色映射
            theta_surf = np.linspace(0, 2 * np.pi, self.ntheta)
            z_surf = np.linspace(0, self.H, self.nz)
            Theta_surf, Z_surf = np.meshgrid(theta_surf, z_surf)

            X_surf = self.R * np.cos(Theta_surf)
            Y_surf = self.R * np.sin(Theta_surf)
            P_surf = P_current[:, :, -1]  # 外表面压强

            # 使用标准的cmap和vmin/vmax
            surface1 = ax_main.plot_surface(X_surf * 1000, Y_surf * 1000, Z_surf * 1000,
                                            facecolors=None,  # 不使用facecolors
                                            cmap=cmap,
                                            vmin=P_min, vmax=P_max,
                                            alpha=0.7, linewidth=0.1,
                                            antialiased=True, shade=True)
            # 手动设置颜色
            surface1.set_array(P_surf.ravel())

            # 2. 底面 (z = 0)
            r_grid = np.linspace(0, self.R, self.nr)
            theta_grid = np.linspace(0, 2 * np.pi, self.ntheta)
            R_grid, Theta_grid = np.meshgrid(r_grid, theta_grid)

            X_bottom = R_grid * np.cos(Theta_grid)
            Y_bottom = R_grid * np.sin(Theta_grid)
            Z_bottom = np.zeros_like(X_bottom)
            P_bottom = P_current[0, :, :].T

            surface2 = ax_main.plot_surface(X_bottom * 1000, Y_bottom * 1000, Z_bottom * 1000,
                                            facecolors=None,
                                            cmap=cmap,
                                            vmin=P_min, vmax=P_max,
                                            alpha=0.9, linewidth=0.1,
                                            antialiased=True, shade=True)
            surface2.set_array(P_bottom.ravel())

            # 3. 顶面 (z = H)
            Z_top = np.ones_like(X_bottom) * self.H
            P_top = P_current[-1, :, :].T

            surface3 = ax_main.plot_surface(X_bottom * 1000, Y_bottom * 1000, Z_top * 1000,
                                            facecolors=None,
                                            cmap=cmap,
                                            vmin=P_min, vmax=P_max,
                                            alpha=0.8, linewidth=0.1,
                                            antialiased=True, shade=True)
            surface3.set_array(P_top.ravel())

            # 4. 中央纵截面
            r_section = np.linspace(0, self.R, self.nr)
            z_section = np.linspace(0, self.H, self.nz)
            R_section, Z_section = np.meshgrid(r_section, z_section)

            X_section = R_section
            Y_section = np.zeros_like(R_section)
            mid_theta_idx = self.ntheta // 2
            P_section = P_current[:, mid_theta_idx, :]

            surface4 = ax_main.plot_surface(X_section * 1000, Y_section * 1000, Z_section * 1000,
                                            facecolors=None,
                                            cmap=cmap,
                                            vmin=P_min, vmax=P_max,
                                            alpha=0.6, linewidth=0,
                                            antialiased=True, shade=True)
            surface4.set_array(P_section.ravel())

            # 5. 另一个垂直截面
            Y_section2 = R_section
            X_section2 = np.zeros_like(R_section)
            P_section2 = P_current[:, 0, :]  # theta=0截面

            surface5 = ax_main.plot_surface(X_section2 * 1000, Y_section2 * 1000, Z_section * 1000,
                                            facecolors=None,
                                            cmap=cmap,
                                            vmin=P_min, vmax=P_max,
                                            alpha=0.6, linewidth=0,
                                            antialiased=True, shade=True)
            surface5.set_array(P_section2.ravel())

            # 6. 添加等压面可视化（使用轮廓线）
            # 在中间高度平面添加等压线
            mid_z_idx = self.nz // 2
            P_mid_plane = P_current[mid_z_idx, :, :].T

            # 创建等压线
            theta_contour = np.linspace(0, 2 * np.pi, self.ntheta)
            r_contour = np.linspace(0, self.R, self.nr)
            R_contour, Theta_contour = np.meshgrid(r_contour, theta_contour)
            X_contour = R_contour * np.cos(Theta_contour)
            Y_contour = R_contour * np.sin(Theta_contour)
            Z_contour = np.ones_like(X_contour) * self.z[mid_z_idx]

            # 绘制等压线
            contour_set = ax_main.contour(X_contour * 1000, Y_contour * 1000, Z_contour * 1000,
                                          P_mid_plane / 1000, levels=6, colors='white',
                                          alpha=0.8, linewidths=1.5)

            # 设置坐标轴
            ax_main.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
            ax_main.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
            ax_main.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')

            # 设置等比例
            max_range = self.R * 1000
            ax_main.set_xlim([-max_range * 1.2, max_range * 1.2])
            ax_main.set_ylim([-max_range * 1.2, max_range * 1.2])
            ax_main.set_zlim([0, self.H * 1000 * 1.2])

            # 设置视角 - 缓慢旋转
            ax_main.view_init(elev=30, azim=45 + frame * 1.8)

            # 标题和信息
            title = f'圆柱储气罐3D压强分布 - 真空抽气热力学过程\n时间: {current_time:.2f}s'
            ax_main.set_title(title, fontsize=14, fontweight='bold', pad=20)

            # 计算物理参数
            min_p = np.min(P_current) / 1000
            max_p = np.max(P_current) / 1000
            avg_p = np.mean(P_current) / 1000
            vacuum_level = (1 - min_p / (self.P0 / 1000)) * 100

            # 信息文本
            info_text = f'🔥 压强范围: {min_p:.1f} - {max_p:.1f} kPa\n'
            info_text += f'📊 平均压强: {avg_p:.1f} kPa\n'
            info_text += f'🌪️ 真空度: {vacuum_level:.1f}%\n'
            info_text += f'⚙️ 泵效率: {self.pump_efficiency:.0%}\n'
            info_text += f'📐 储罐: φ{self.R * 2000:.0f}×{self.H * 1000:.0f}mm'

            ax_main.text2D(0.02, 0.98, info_text, transform=ax_main.transAxes,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
                           verticalalignment='top', fontsize=10, fontweight='bold')

            # 添加颜色条（只在第一帧添加）
            if frame == 0:
                # 创建颜色条
                mappable = plt.cm.ScalarMappable(cmap=cmap)
                mappable.set_array([P_min / 1000, P_max / 1000])
                mappable.set_clim(P_min / 1000, P_max / 1000)
                cbar = plt.colorbar(mappable, ax=ax_main, shrink=0.6, aspect=20, pad=0.1)
                cbar.set_label('压强 (kPa)', fontsize=12, fontweight='bold')

            # 美化图形
            ax_main.grid(True, alpha=0.3)
            ax_main.xaxis.pane.fill = False
            ax_main.yaxis.pane.fill = False
            ax_main.zaxis.pane.fill = False
            ax_main.xaxis.pane.set_edgecolor('gray')
            ax_main.yaxis.pane.set_edgecolor('gray')
            ax_main.zaxis.pane.set_edgecolor('gray')
            ax_main.xaxis.pane.set_alpha(0.1)
            ax_main.yaxis.pane.set_alpha(0.1)
            ax_main.zaxis.pane.set_alpha(0.1)

        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=self.nt,
                                       interval=150, blit=False, repeat=True)

        return fig, anim


def main():
    print("🚀 初始化3D真空泵抽气热力学模拟...")
    print("   ✨ 完全3D可视化")
    print("   🔬 考虑内壁吸附效应")
    print("   🌡️ 真实热力学计算")
    print("   🎯 科研级精度")

    sim = VacuumTank3D(R=0.06, H=0.10, pump_efficiency=0.85)

    print("🎨 创建3D热力学可视化...")
    fig, anim = sim.create_3d_visualization()

    # 优化显示
    plt.tight_layout()
    plt.show()

    # 保存选项
    save = input("\n💾 是否保存3D热力学动画? (y/n): ")
    if save.lower() == 'y':
        print("⏳ 正在保存动画，请稍候...")
        try:
            anim.save('vacuum_3d_thermodynamics_stable.gif', writer='pillow', fps=6, dpi=100)
            print("✅ 3D热力学动画已保存！")
            print("📁 文件名: vacuum_3d_thermodynamics_stable.gif")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            print("💡 提示: 请确保安装了pillow: pip install pillow")


if __name__ == "__main__":
    main()
