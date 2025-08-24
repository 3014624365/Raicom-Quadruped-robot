import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class PolynomialRegressionComparison:
    def __init__(self):
        # 原始数据
        self.data = [
            (5, 1, 0.5), (5, 2, 5.2), (5, 3, 9.8), (5, 4, 14.3), (5, 5, 18.4),
            (10, 1, 3.7), (10, 2, 14), (10, 3, 25.2), (10, 4, 36.2),
            (7, 1, 1.7), (7, 2, 8.8), (7, 3, 16.9), (7, 4, 24.1), (7, 5, 30.8),
            (3, 1, -0.6), (3, 2, 2.6), (3, 3, 4.8), (3, 4, 5.6), (3, 5, 8.3), (3, 6, 10.6),
            (-5, 1, -4.5), (-5, 2, -12.1), (-5, 3, -19), (-5, 4, -26.7), (-5, 5, -33.4),
            (-3, 1, -3.2), (-3, 2, -8.5), (-3, 3, -13.6), (-3, 4, -18.5), (-3, 5, -22.5), (-3, 6, -27.2), (-3, 7, -34),
            (-6, 1, -5.6), (-6, 2, -15), (-6, 3, -24), (-6, 4, -34.2), (-6, 5, -42.5), (-6, 6, -53),
            (-7, 1, -6.3), (-7, 2, -18.1), (-7, 3, -29), (-7, 4, -39.5), (-7, 5, -51),
            (-8, 1, -7.2), (-8, 2, -20), (-8, 3, -33), (-8, 4, -45.5),
            (-9, 1, -8.4), (-9, 2, -23.2), (-9, 3, -36.5), (-9, 4, -52.5),
            (-10, 1, -9.4), (-10, 2, -26), (-10, 3, -42), (-10, 4, -57.5)
        ]

        # 转换为numpy数组
        self.vx_values = np.array([d[0] for d in self.data])
        self.time_values = np.array([d[1] for d in self.data])
        self.distance_values = np.array([d[2] for d in self.data])

        self.models = {}
        self.poly_transformers = {}
        self.results = {}

    def fit_polynomial_models(self):
        """拟合2次、3次、4次多项式模型"""
        print("拟合多项式回归模型...")
        print("=" * 80)

        X_base = np.column_stack([self.vx_values, self.time_values])

        for degree in [2, 3, 4]:
            print(f"\n拟合 {degree} 次多项式回归:")

            # 创建多项式特征
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly.fit_transform(X_base)

            # 拟合模型
            model = LinearRegression(fit_intercept=False)  # 因为PolynomialFeatures已包含偏置项
            model.fit(X_poly, self.distance_values)

            # 存储模型和变换器
            self.models[f'Poly_{degree}'] = model
            self.poly_transformers[f'Poly_{degree}'] = poly

            # 计算预测值和评估指标
            y_pred = model.predict(X_poly)
            r2 = r2_score(self.distance_values, y_pred)
            rmse = np.sqrt(mean_squared_error(self.distance_values, y_pred))
            mae = mean_absolute_error(self.distance_values, y_pred)

            self.results[f'Poly_{degree}'] = {
                'predictions': y_pred,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'degree': degree,
                'n_features': X_poly.shape[1]
            }

            print(f"  R² Score: {r2:.8f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  特征数量: {X_poly.shape[1]}")

            # 输出详细的公式系数
            self.print_formula_coefficients(degree, model, poly)

    def print_formula_coefficients(self, degree, model, poly):
        """输出详细的公式系数"""
        print(f"\n  {degree}次多项式公式:")

        # 获取特征名称
        feature_names = poly.get_feature_names_out(['vx', 'time'])
        coefficients = model.coef_

        # 构建公式字符串
        formula_parts = []

        for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
            if abs(coef) < 1e-10:  # 忽略非常小的系数
                continue

            # 格式化系数
            if i == 0:  # 第一项
                if feature == '1':  # 常数项
                    formula_parts.append(f"{coef:.8f}")
                else:
                    formula_parts.append(f"{coef:.8f}*{feature}")
            else:
                if coef >= 0:
                    if feature == '1':
                        formula_parts.append(f" + {coef:.8f}")
                    else:
                        formula_parts.append(f" + {coef:.8f}*{feature}")
                else:
                    if feature == '1':
                        formula_parts.append(f" - {abs(coef):.8f}")
                    else:
                        formula_parts.append(f" - {abs(coef):.8f}*{feature}")

        formula = "distance = " + "".join(formula_parts)

        # 替换特征名称为更易读的形式
        formula = formula.replace('vx^2', 'vx²')
        formula = formula.replace('time^2', 'time²')
        formula = formula.replace('vx^3', 'vx³')
        formula = formula.replace('time^3', 'time³')
        formula = formula.replace('vx^4', 'vx⁴')
        formula = formula.replace('time^4', 'time⁴')

        print(f"  {formula}")

        # 分别输出各项系数
        print(f"  \n  系数详情:")
        for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
            if abs(coef) > 1e-10:
                print(f"    {feature}: {coef:.8f}")

    def predict_with_polynomial(self, degree, vx, time):
        """使用指定次数的多项式模型进行预测"""
        model = self.models[f'Poly_{degree}']
        poly = self.poly_transformers[f'Poly_{degree}']

        X_input = np.array([[vx, time]])
        X_poly = poly.transform(X_input)

        return model.predict(X_poly)[0]

    def create_comparison_table(self):
        """创建模型比较表"""
        print("\n\n" + "=" * 100)
        print("多项式回归模型比较表")
        print("=" * 100)

        df_data = []
        for name, result in self.results.items():
            df_data.append({
                '模型': name,
                '次数': result['degree'],
                'R² Score': result['r2'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                '特征数量': result['n_features']
            })

        df = pd.DataFrame(df_data)
        print(df.to_string(index=False, float_format='%.8f'))

        # 找出最佳模型
        best_model = df.loc[df['R² Score'].idxmax()]
        print(f"\n最佳模型: {best_model['模型']} (R² = {best_model['R² Score']:.8f})")

    def visualize_comparison(self):
        """可视化比较结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. R²比较
        degrees = [2, 3, 4]
        r2_scores = [self.results[f'Poly_{d}']['r2'] for d in degrees]

        axes[0, 0].bar([f'{d}次' for d in degrees], r2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('R² Score 比较')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].grid(True, alpha=0.3)
        for i, score in enumerate(r2_scores):
            axes[0, 0].text(i, score + 0.001, f'{score:.6f}', ha='center', va='bottom')

        # 2. RMSE比较
        rmse_scores = [self.results[f'Poly_{d}']['rmse'] for d in degrees]
        axes[0, 1].bar([f'{d}次' for d in degrees], rmse_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 1].set_title('RMSE 比较')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].grid(True, alpha=0.3)
        for i, score in enumerate(rmse_scores):
            axes[0, 1].text(i, score + 0.05, f'{score:.4f}', ha='center', va='bottom')

        # 3. MAE比较
        mae_scores = [self.results[f'Poly_{d}']['mae'] for d in degrees]
        axes[0, 2].bar([f'{d}次' for d in degrees], mae_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 2].set_title('MAE 比较')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].grid(True, alpha=0.3)
        for i, score in enumerate(mae_scores):
            axes[0, 2].text(i, score + 0.03, f'{score:.4f}', ha='center', va='bottom')

        # 4. 预测vs实际对比（所有模型）
        colors = ['blue', 'green', 'red']
        for i, degree in enumerate(degrees):
            predictions = self.results[f'Poly_{degree}']['predictions']
            axes[1, 0].scatter(self.distance_values, predictions, alpha=0.6,
                               color=colors[i], label=f'{degree}次多项式')

        # 添加理想线
        min_val, max_val = self.distance_values.min(), self.distance_values.max()
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)
        axes[1, 0].set_xlabel('实际距离')
        axes[1, 0].set_ylabel('预测距离')
        axes[1, 0].set_title('预测 vs 实际值比较')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. 残差分析
        best_degree = max(degrees, key=lambda d: self.results[f'Poly_{d}']['r2'])
        best_predictions = self.results[f'Poly_{best_degree}']['predictions']
        residuals = self.distance_values - best_predictions

        axes[1, 1].scatter(best_predictions, residuals, alpha=0.7, color='purple')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[1, 1].set_xlabel('预测距离')
        axes[1, 1].set_ylabel('残差')
        axes[1, 1].set_title(f'最佳模型({best_degree}次) 残差分析')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. 特征数量vs性能
        n_features = [self.results[f'Poly_{d}']['n_features'] for d in degrees]
        axes[1, 2].plot(n_features, r2_scores, 'o-', color='darkblue', linewidth=2, markersize=8)
        axes[1, 2].set_xlabel('特征数量')
        axes[1, 2].set_ylabel('R² Score')
        axes[1, 2].set_title('模型复杂度 vs 性能')
        axes[1, 2].grid(True, alpha=0.3)

        # 标注每个点
        for i, (x, y) in enumerate(zip(n_features, r2_scores)):
            axes[1, 2].annotate(f'{degrees[i]}次', (x, y), textcoords="offset points",
                                xytext=(0, 10), ha='center')

        plt.tight_layout()
        plt.show()

    def visualize_3d_surfaces(self):
        """3D可视化不同次数多项式的拟合表面"""
        fig = plt.figure(figsize=(20, 6))

        # 创建网格用于绘制表面
        vx_range = np.linspace(self.vx_values.min(), self.vx_values.max(), 30)
        time_range = np.linspace(self.time_values.min(), self.time_values.max(), 30)
        VX, TIME = np.meshgrid(vx_range, time_range)

        for i, degree in enumerate([2, 3, 4]):
            ax = fig.add_subplot(1, 3, i + 1, projection='3d')

            # 计算预测表面
            DIST_PRED = np.zeros_like(VX)
            for row in range(VX.shape[0]):
                for col in range(VX.shape[1]):
                    DIST_PRED[row, col] = self.predict_with_polynomial(degree, VX[row, col], TIME[row, col])

            # 绘制表面
            surface = ax.plot_surface(VX, TIME, DIST_PRED, alpha=0.7, cmap='viridis')

            # 添加原始数据点
            ax.scatter(self.vx_values, self.time_values, self.distance_values,
                       color='red', s=30, alpha=1.0, label='实际数据')

            ax.set_xlabel('速度 (vx)')
            ax.set_ylabel('时间 (time)')
            ax.set_zlabel('距离 (distance)')
            ax.set_title(f'{degree}次多项式拟合表面\nR² = {self.results[f"Poly_{degree}"]["r2"]:.6f}')

        plt.tight_layout()
        plt.show()

    def test_predictions(self):
        """测试几个预测例子"""
        print("\n" + "=" * 80)
        print("预测测试")
        print("=" * 80)

        test_cases = [
            (5, 3, "中等正速度"),
            (-7, 2, "中等负速度"),
            (10, 1, "高正速度短时间"),
            (-10, 4, "高负速度长时间")
        ]

        print(f"{'测试情况':<15} {'vx':<5} {'time':<5} {'2次预测':<10} {'3次预测':<10} {'4次预测':<10}")
        print("-" * 80)

        for vx, time, description in test_cases:
            pred_2 = self.predict_with_polynomial(2, vx, time)
            pred_3 = self.predict_with_polynomial(3, vx, time)
            pred_4 = self.predict_with_polynomial(4, vx, time)

            print(f"{description:<15} {vx:<5} {time:<5} {pred_2:<10.4f} {pred_3:<10.4f} {pred_4:<10.4f}")


def main():
    # 创建比较实例
    poly_comparison = PolynomialRegressionComparison()

    # 拟合多项式模型
    poly_comparison.fit_polynomial_models()

    # 创建比较表
    poly_comparison.create_comparison_table()

    # 可视化比较
    poly_comparison.visualize_comparison()

    # 3D表面可视化
    poly_comparison.visualize_3d_surfaces()

    # 测试预测
    poly_comparison.test_predictions()

    return poly_comparison


# 运行比较
if __name__ == "__main__":
    poly_comparison = main()
