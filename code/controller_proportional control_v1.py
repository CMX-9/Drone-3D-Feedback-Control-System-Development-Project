def controller(state, target_pos, dt):
    # 状态格式: [x, y, z, roll, pitch, yaw]
    # 目标格式: (x, y, z, yaw)
    # dt: 时间步长
    # 返回值: (vx, vy, vz, yaw_rate)

    # ====== 状态持久化初始化 ======
    if not hasattr(controller, "storage"):
        # 轨迹存储结构
        controller.storage = {
            "current_target": None,
            "trajectory": [],
            "step": 0,
            "last_yaw": 0.0
        }

    # ====== 轨迹管理逻辑 ======
    # 检测目标变化(使用坐标容差0.1米，角度容差5度)
    target_changed = (
        controller.storage["current_target"] is None or
        any(abs(t - s) > 0.1 for t, s in zip(target_pos[:3], controller.storage["current_target"][:3])) or
        abs(target_pos[3] - controller.storage["current_target"][3]) > 0.087  # 5度
    )

    if target_changed:
        # ====== 轨迹生成核心 ======
        def local_min_jerk(current, target, dt, duration=3.0):
            """局部最小加加速度轨迹生成器"""
            num = int(round(duration/dt))
            return [current + (target - current) * (10*(i/num)**3 -15*(i/num)**4 +6*(i/num)**5)
                    for i in range(num)]

        # 生成各轴轨迹
        x_traj = local_min_jerk(state[0], target_pos[0], dt)
        y_traj = local_min_jerk(state[1], target_pos[1], dt)
        z_traj = local_min_jerk(state[2], target_pos[2], dt)
        yaw_traj = local_min_jerk(state[5], target_pos[3], dt)

        # 轨迹对齐(取最小长度)
        min_len = min(len(x_traj), len(y_traj), len(z_traj), len(yaw_traj))
        controller.storage["trajectory"] = list(zip(
            x_traj[:min_len],
            y_traj[:min_len],
            z_traj[:min_len],
            yaw_traj[:min_len]
        ))
        controller.storage["step"] = 0
        controller.storage["current_target"] = target_pos

    # ====== 轨迹跟踪逻辑 ======
    if controller.storage["step"] < len(controller.storage["trajectory"]):
        target = controller.storage["trajectory"][controller.storage["step"]]
        controller.storage["step"] += 1
    else:
        target = target_pos  # 到达后保持最终目标

    # ====== 坐标系转换 ======
    current_yaw = state[5]
    dx = target[0] - state[0]
    dy = target[1] - state[1]
    
    # 手动实现旋转矩阵(泰勒展开近似)
    cos_y = 1.0 - current_yaw**2/2 + current_yaw**4/24  # cos近似
    sin_y = current_yaw - current_yaw**3/6 + current_yaw**5/120  # sin近似
    
    # 机体坐标系误差
    error_x = dx * cos_y + dy * sin_y
    error_y = -dx * sin_y + dy * cos_y
    error_z = target[2] - state[2]
    
    # 偏航角误差处理(手动角度包裹)
    yaw_error = target[3] - current_yaw
    yaw_error = (yaw_error + 3.1415926535) % 6.283185307 - 3.1415926535

    # ====== 控制律计算 ======
    # 比例增益
    Kp_xy = 0.8
    Kp_z = 1.2
    Kp_yaw = 0.6

    vx = Kp_xy * error_x
    vy = Kp_xy * error_y
    vz = Kp_z * error_z
    yaw_rate = Kp_yaw * yaw_error

    # ====== 输出限幅 ======
    vx = max(min(vx, 1.0), -1.0)
    vy = max(min(vy, 1.0), -1.0)
    vz = max(min(vz, 0.5), -0.5)
    yaw_rate = max(min(yaw_rate, 1.0), -1.0)

    return (vx, vy, vz, yaw_rate)