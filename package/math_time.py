import numpy as np
from scipy.optimize import fsolve


def calculate_dog_runtime(target_distance, vx):
    """
    根据目标距离和速度计算机器狗运行时间
    基于4次多项式拟合模型 (R² = 99.91%)

    Args:
        target_distance: 目标行走距离
        vx: 移动速度

    Returns:
        float: 所需运行时间(秒)，失败返回None
    """

    # 4次多项式系数
    coefficients = np.array([
        0.14742973,  # 常数项
        -0.67239493,  # vx
        -1.91332420,  # time
        -0.00673316,  # vx²
        1.22802422,  # vx × time
        0.21743086,  # time²
        -0.00121133,  # vx³
        -0.00915058,  # vx² × time
        -0.01756123,  # vx × time²
        -0.00951675,  # time³
        0.00001878,  # vx⁴
        0.00207478,  # vx³ × time
        -0.00117696,  # vx² × time²
        0.00164439,  # vx × time³
        -0.00157226  # time⁴
    ])

    def predict_distance(vx, time):
        """预测距离的内部函数"""
        features = np.array([
            1, vx, time, vx ** 2, vx * time, time ** 2, vx ** 3,
                         vx ** 2 * time, vx * time ** 2, time ** 3, vx ** 4,
                         vx ** 3 * time, vx ** 2 * time ** 2, vx * time ** 3, time ** 4
        ])
        return np.dot(coefficients, features)

    # 输入验证
    if abs(vx) < 0.01:
        return None

    # 智能初始猜测
    if vx > 0:
        initial_guess = max(0.5, target_distance / (vx * 3))
    else:
        initial_guess = max(0.5, abs(target_distance) / (abs(vx) * 3))
    initial_guess = min(initial_guess, 7.5)  # 限制在合理范围

    # 目标函数
    def objective(time_array):
        time = time_array[0]
        if time <= 0:
            return 1e10
        return predict_distance(vx, time) - target_distance

    # 求解
    try:
        solution = fsolve(objective, [initial_guess])
        runtime = solution[0]

        # 验证解的有效性
        if 0.1 <= runtime <= 15.0:
            predicted_dist = predict_distance(vx, runtime)
            error = abs(predicted_dist - target_distance)

            if error < 0.1:  # 误差小于0.1
                return runtime

        return None

    except:
        return None


# 使用示例
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        (10.0, 5),  # 目标距离10，速度5
        (-15.0, -4),  # 目标距离-15，速度-4
        (25.0, 8),  # 目标距离25，速度8
        (-30.0, -7),  # 目标距离-30，速度-7
    ]

    print("机器狗运行时间计算测试:")
    print("-" * 50)
    for target_dist, speed in test_cases:
        runtime = calculate_dog_runtime(target_dist, speed)
        if runtime:
            print(f"目标距离: {target_dist:6.1f}, 速度: {speed:3}, 运行时间: {runtime:.4f}秒")
        else:
            print(f"目标距离: {target_dist:6.1f}, 速度: {speed:3}, 计算失败")
