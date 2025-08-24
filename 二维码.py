self.adjustment_history.append(current_distance)
# 计算误差
error = current_distance - self.target_distance
print(f"当前距离: {current_distance:.1f} mm, 误差: {error:.1f} mm")
# 根据误差调整位置
if error > 200:
    print("大误差，显著后退")
    self.adjust_x(-5, 3)
elif error > 100:
    print("中等误差，适度后退")
    self.adjust_x(-5, 1.5)
elif error > 50:
    print("小误差，轻微后退")
    self.adjust_x(-5, 1)
elif error > 10:
    print("微调，小幅后退")
    self.adjust_x(-2, 1)
elif error < -200:
    print("大误差，显著前进")
    self.adjust_x(5, 3)
elif error < -100:
    print("中等误差，适度前进")
    self.adjust_x(5, 1.5)
elif error < -50:
    print("小误差，轻微前进")
    self.adjust_x(5, 1)
elif error < -10:
    print("微调，小幅前进")
    self.adjust_x(2, 1)
else:
    # 达到目标
    print(f"达到目标距离: {current_distance:.1f} mm")
    adjustment_success = True
    break