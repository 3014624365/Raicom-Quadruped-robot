import time
from xgolib import XGO
# 增强型自适应陀螺仪控制器，修复角度校正方向问题
class AdaptiveYawController:
    """自适应陀螺仪控制器，带惯性补偿，修复角度方向校正"""

    def __init__(self, dog):
        self.dog = dog
        # 转向校正系数 (实测值/理论值)
        self.left_correction = 1.40  # 左转校正
        self.right_correction = 1.40  # 右转校正（已统一为左转相同的系数）
        self.inertia_wait_time = 0.8  # 惯性稳定等待时间（秒）

        # 确保机器狗在初始化时稳定
        print("等待初始稳定...")
        time.sleep(1.0)

        # 获取稳定的初始参考角度，存储为绝对角度和标准化角度两种形式
        self.reference_yaw_absolute = self.get_stable_yaw(readings=7)  # 多次读取平均 - 原始累积值
        self.reference_yaw = self.reference_yaw_absolute % 360  # 标准化到0-360度

        print(f"创建了自适应陀螺仪控制器 (转向校正: {self.left_correction})")
        print(f"初始参考角度: 原始={self.reference_yaw_absolute}°, 标准化={self.reference_yaw:.1f}°")

        # 用于记录校正过程的调试信息
        self.debug_logs = []

    def log_debug(self, message):
        """记录调试信息"""
        print(f"[DEBUG] {message}")
        self.debug_logs.append(message)

    def get_stable_yaw(self, readings=5, interval=0.05, normalize=False):
        """
        获取多次稳定的陀螺仪读数平均值

        参数:
            readings: 读取次数
            interval: 读取间隔时间
            normalize: 是否返回标准化到0-360度的值
        """
        readings_list = []
        for _ in range(readings):
            readings_list.append(self.dog.read_yaw())
            time.sleep(interval)

        # 计算平均值和标准差
        avg_yaw = sum(readings_list) / len(readings_list)
        variance = sum((x - avg_yaw) ** 2 for x in readings_list) / len(readings_list)
        std_dev = variance ** 0.5

        # 如果标准差大于2度，说明读数不稳定，增加采样次数
        if std_dev > 2.0 and readings < 10:
            print(f"陀螺仪读数不稳定(标准差={std_dev:.2f}°)，增加采样")
            return self.get_stable_yaw(readings=readings + 2, interval=interval, normalize=normalize)

        # 根据需要返回标准化值或原始值
        if normalize:
            return avg_yaw % 360
        else:
            return avg_yaw

    def normalize_angle(self, angle):
        """将角度标准化到-180到+180度范围内"""
        angle = angle % 360  # 先标准化到0-360度
        if angle > 180:
            angle -= 360
        return angle

    def calculate_angle_diff(self, current, target):
        """
        计算两个角度之间的最短角度差（考虑360度循环）

        参数:
            current: 当前角度 (0-360范围内)
            target: 目标角度 (0-360范围内)
        """
        # 确保输入角度在0-360范围内
        current_norm = current % 360
        target_norm = target % 360

        # 计算原始差值
        diff = target_norm - current_norm

        # 选择最短路径
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360

        return diff

    def turn_by_angle(self, angle_change, speed=12, timeout=15):  # 降低默认速度
        """
        执行相对角度的转向，使用校正系数补偿系统偏差，并消除惯性

        参数:
            angle_change: 要转向的角度（正值左转，负值右转）
            speed: 转向速度 (已降低)
            timeout: 最大执行时间（秒）
        """
        # 标准化角度到-180到+180范围
        angle_change = self.normalize_angle(angle_change)

        # 记录起始角度 - 使用稳定读数
        print("获取稳定的起始角度...")
        start_angle = self.get_stable_yaw()
        start_angle_norm = start_angle % 360
        print(f"开始自适应转向: 当前角度={start_angle}° (标准化={start_angle_norm:.1f}°), 目标变化={angle_change}°")

        # 计算目标绝对角度
        target_angle = start_angle + angle_change
        target_angle_norm = target_angle % 360
        print(f"目标绝对角度: {target_angle}° (标准化={target_angle_norm:.1f}°)")

        # 应用校正系数
        corrected_angle = angle_change
        if angle_change > 0:  # 左转
            corrected_angle = angle_change * self.left_correction
        else:  # 右转
            corrected_angle = angle_change * self.right_correction

        print(f"应用校正系数后的目标角度变化: {corrected_angle}°")

        # 目标是达到的绝对角度差
        target_diff = abs(angle_change)  # 使用原始目标进行验证

        # 设置转向方向和速度 - 降低速度防止惯性过大
        turn_direction = 1 if corrected_angle > 0 else -1  # 1为左转，-1为右转
        turn_speed = min(abs(speed), 30)  # 进一步限制最大速度

        # 估计转向时间（基于校正后的角度）- 由于速度降低，每度角需要更长时间
        estimated_time = abs(corrected_angle) * 0.04 + 0.6  # 调整系数，适应较低速度
        print(f"估计转向时间: {estimated_time:.2f}秒")

        # 提前停止位置的百分比 - 用于抵消惯性
        inertia_factor = 0.85  # 提前到85%位置停止，让惯性带到目标位置

        # 转向第一阶段：主要转向
        main_time = estimated_time * inertia_factor
        print(f"开始转向，速度={turn_speed * turn_direction}，转向时间={main_time:.2f}秒")
        self.dog.turn(turn_speed * turn_direction)
        time.sleep(main_time)

        # 停止并等待惯性稳定
        print("转向停止，等待惯性稳定")
        self.dog.turn(0)
        time.sleep(self.inertia_wait_time)  # 等待惯性稳定

        # 获取稳定的当前角度
        current_angle = self.get_stable_yaw()
        current_angle_norm = current_angle % 360

        # 计算到目标的剩余角度（考虑360度循环）
        # 使用标准化值计算
        remaining_angle = self.calculate_angle_diff(current_angle_norm, target_angle_norm)
        print(f"当前角度: {current_angle}° (标准化={current_angle_norm:.1f}°)")
        print(f"目标角度: {target_angle}° (标准化={target_angle_norm:.1f}°)")
        print(f"剩余需要转向: {remaining_angle:.1f}°")

        # 如果剩余角度较小，使用微调来完成最后的调整
        if abs(remaining_angle) > 3 and abs(remaining_angle) < 30:
            print(f"开始微调剩余角度: {remaining_angle:.1f}°")

            # 计算微调方向和速度
            fine_direction = 1 if remaining_angle > 0 else -1
            fine_speed = min(abs(remaining_angle) * 0.5, 10)  # 速度与剩余角度成正比，但不超过10

            # 微调时间计算
            fine_time = abs(remaining_angle) * 0.05 + 0.3  # 微调时间

            print(f"开始微调，速度={fine_speed * fine_direction}，微调时间={fine_time:.2f}秒")
            self.dog.turn(fine_speed * fine_direction)
            time.sleep(fine_time)

            # 停止并等待稳定
            self.dog.turn(0)
            time.sleep(self.inertia_wait_time)  # 等待惯性稳定

        # 验证最终角度变化
        end_angle = self.get_stable_yaw()
        end_angle_norm = end_angle % 360

        # 计算实际角度变化
        # 考虑使用标准化值或绝对值
        if abs(end_angle - start_angle) < 180:  # 如果绝对差值较小，使用绝对值
            actual_change = end_angle - start_angle
        else:  # 否则使用标准化值计算
            actual_change = self.calculate_angle_diff(start_angle_norm, end_angle_norm)

        print(f"最终角度变化: {actual_change:.1f}° (目标: {angle_change}°)")
        print(f"结束角度: {end_angle}° (标准化={end_angle_norm:.1f}°)")

        # 计算准确率
        accuracy = abs(actual_change) / target_diff if target_diff != 0 else 1.0
        print(f"转向准确率: {accuracy:.2f}")

        # 判断是否成功 - 允许20%的误差
        success = abs(actual_change - angle_change) < abs(angle_change) * 0.2
        print(f"转向{'成功' if success else '失败'}")

        return success, actual_change

    def turn_to_target_yaw(self, target_yaw, speed=12, error_threshold=5):
        """
        转向到指定的绝对偏航角

        参数:
            target_yaw: 目标绝对偏航角
            speed: 转向速度
            error_threshold: 允许的误差范围（度）
        """
        # 获取当前稳定角度
        current_yaw = self.get_stable_yaw()
        current_yaw_norm = current_yaw % 360

        # 标准化目标角度
        target_yaw_norm = target_yaw % 360

        # 计算需要转动的角度（考虑360度循环）
        angle_diff = self.calculate_angle_diff(current_yaw_norm, target_yaw_norm)

        print(f"转向到目标偏航角: 当前={current_yaw}° (标准化={current_yaw_norm:.1f}°)")
        print(f"目标={target_yaw}° (标准化={target_yaw_norm:.1f}°), 差值={angle_diff:.1f}°")
        self.log_debug(f"转向到目标: 当前={current_yaw_norm:.1f}°, 目标={target_yaw_norm:.1f}°, 差值={angle_diff:.1f}°")

        # 如果已经在目标范围内，无需转向
        if abs(angle_diff) <= error_threshold:
            print("已在目标角度范围内，无需转向")
            return True, 0

        # 执行转向
        return self.turn_by_angle(angle_diff, speed)

    def check_and_correct_orientation(self, expected_yaw=None, error_threshold=8, max_attempts=2):
        """
        检查当前角度是否偏离预期，如果偏离则尝试校正

        参数:
            expected_yaw: 预期的偏航角，如果为None则使用初始参考角度
            error_threshold: 允许的误差范围（度）- 已增加
            max_attempts: 最大尝试校正次数
        """
        # 如果没有指定预期角度，使用参考角度
        if expected_yaw is None:
            expected_yaw = self.reference_yaw_absolute

        # 获取当前稳定角度
        current_yaw = self.get_stable_yaw()

        # 标准化角度进行比较 - 仅关注0-360度范围
        current_yaw_norm = current_yaw % 360
        expected_yaw_norm = expected_yaw % 360

        # 计算角度差（使用标准化值）
        angle_diff = self.calculate_angle_diff(current_yaw_norm, expected_yaw_norm)

        self.log_debug(f"校正检查: 当前={current_yaw}° (标准化={current_yaw_norm:.1f}°), " +
                       f"预期={expected_yaw}° (标准化={expected_yaw_norm:.1f}°), 差值={angle_diff:.1f}°")

        # 如果在允许范围内，无需校正 - 使用更宽松的阈值
        if abs(angle_diff) <= error_threshold:
            print(f"当前角度={current_yaw_norm:.1f}°, 预期角度={expected_yaw_norm:.1f}°, " +
                  f"误差={angle_diff:.1f}°, 在允许范围内(阈值±{error_threshold}°)")
            return True

        # 需要校正
        print(f"检测到角度偏离: 当前={current_yaw_norm:.1f}°, 预期={expected_yaw_norm:.1f}°, " +
              f"误差={angle_diff:.1f}°, 超出阈值(±{error_threshold}°)")

        # 尝试校正 - 基于标准化角度差值
        for attempt in range(max_attempts):
            print(f"尝试校正角度 (尝试 {attempt + 1}/{max_attempts})")
            # 使用相对转向基于标准化角度差值
            success, _ = self.turn_by_angle(angle_diff, speed=10)

            if success:
                print("角度校正成功")
                return True

            # 重新获取当前状态
            current_yaw = self.get_stable_yaw()
            current_yaw_norm = current_yaw % 360
            angle_diff = self.calculate_angle_diff(current_yaw_norm, expected_yaw_norm)

            # 再次检查是否在允许范围内
            if abs(angle_diff) <= error_threshold:
                print(f"经过尝试后，角度已在允许范围内(当前={current_yaw_norm:.1f}°, 目标={expected_yaw_norm:.1f}°)")
                return True

        print("角度校正失败，继续执行")
        return False

    def set_reference_yaw(self, new_reference=None):
        """设置新的参考角度"""
        if new_reference is None:
            self.reference_yaw_absolute = self.get_stable_yaw()  # 使用稳定角度
            self.reference_yaw = self.reference_yaw_absolute % 360  # 标准化到0-360度
        else:
            self.reference_yaw_absolute = new_reference
            self.reference_yaw = new_reference % 360  # 标准化到0-360度

        print(f"设置新的参考角度: 原始={self.reference_yaw_absolute}°, 标准化={self.reference_yaw:.1f}°")

#
#
# if __name__ == '__main__':



