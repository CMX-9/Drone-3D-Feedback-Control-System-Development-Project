def controller(state, target_pos, dt):
    # 状态格式: [x, y, z, roll, pitch, yaw] (单位：米，弧度)
    # 目标格式: (x, y, z, yaw) (单位：米，弧度)
    # dt: 时间步长 (秒)
    # 返回值: (vx, vy, vz, yaw_rate) (单位：米/秒，弧度/秒)

    # ====== PID参数优化 ======
    PID_params = {
        'xy': {'Kp': 0.6, 'Ki': 0.03, 'Kd': 0.4, 'Imax': 1.5, 'D_filter': 0.2},  # 水平面参数：降低Kp和Ki，增加D滤波
        'z':  {'Kp': 1.0, 'Ki': 0.08, 'Kd': 0.6, 'Imax': 0.8},  # 高度参数：减小积分限幅
        'yaw':{'Kp': 0.8, 'Ki': 0.01, 'Kd': 0.3, 'Imax': 0.8}   # 偏航参数：提高Kp，降低Ki
    }

    # ====== 持久化状态初始化 ======
    if not hasattr(controller, "storage"):
        controller.storage = {
            'current_target': None,
            'trajectory': [],
            'step': 0,
            
            # PID状态
            'integral_xy': [0.0, 0.0],
            'integral_z': 0.0,
            'integral_yaw': 0.0,
            'prev_error_xy': [0.0, 0.0],
            'prev_error_z': 0.0,
            'prev_error_yaw': 0.0,
            'prev_deriv_xy': [0.0, 0.0],  # 新增：存储滤波后的微分项
            'prev_deriv_yaw': 0.0
        }

    # ====== 轨迹管理优化 ======
    target_changed = (
        controller.storage['current_target'] is None or
        any(abs(t - s) > 0.1 for t, s in zip(target_pos[:3], controller.storage['current_target'][:3])) or
        abs(target_pos[3] - controller.storage['current_target'][3]) > 0.087
    )

    if target_changed:
        # 重置PID状态
        controller.storage.update({
            'integral_xy': [0.0, 0.0],
            'integral_z': 0.0,
            'integral_yaw': 0.0,
            'prev_error_xy': [0.0, 0.0],
            'prev_error_z': 0.0,
            'prev_error_yaw': 0.0,
            'prev_deriv_xy': [0.0, 0.0],
            'prev_deriv_yaw': 0.0
        })

        # 轨迹生成优化：增加轨迹持续时间至4秒
        def generate_trajectory(start, end, dt, duration=4.0):
            num = max(1, int(round(duration/dt)))
            return [start + (end - start) * (10*(i/num)**3 -15*(i/num)**4 +6*(i/num)**5)
                    for i in range(num)]

        # 生成各轴轨迹
        x_traj = generate_trajectory(state[0], target_pos[0], dt)
        y_traj = generate_trajectory(state[1], target_pos[1], dt)
        z_traj = generate_trajectory(state[2], target_pos[2], dt)
        yaw_traj = generate_trajectory(state[5], target_pos[3], dt)

        # 对齐轨迹
        min_len = min(len(x_traj), len(y_traj), len(z_traj), len(yaw_traj))
        controller.storage['trajectory'] = list(zip(x_traj[:min_len], y_traj[:min_len], z_traj[:min_len], yaw_traj[:min_len]))
        controller.storage['step'] = 0
        controller.storage['current_target'] = target_pos

    # ====== 获取当前目标点 ======
    if controller.storage['step'] < len(controller.storage['trajectory']):
        target = controller.storage['trajectory'][controller.storage['step']]
        controller.storage['step'] += 1
    else:
        target = target_pos

    # ====== 坐标系转换优化 ======
    current_yaw = state[5]
    dx = target[0] - state[0]
    dy = target[1] - state[1]
    
    # 使用泰勒展开至五次项提高精度
    cos_y = 1.0 - current_yaw**2/2 + current_yaw**4/24 - current_yaw**6/720  # 添加更高阶项
    sin_y = current_yaw - current_yaw**3/6 + current_yaw**5/120 - current_yaw**7/5040
    
    error_x = dx * cos_y + dy * sin_y
    error_y = -dx * sin_y + dy * cos_y
    error_z = target[2] - state[2]
    yaw_error = (target[3] - current_yaw + 3.1415926535) % 6.283185307 - 3.1415926535

    # ====== PID计算优化 ======
    # X轴PID（带微分滤波）
    raw_deriv_x = (error_x - controller.storage['prev_error_xy'][0]) / dt if dt > 1e-5 else 0
    filtered_deriv_x = PID_params['xy']['D_filter'] * raw_deriv_x + (1 - PID_params['xy']['D_filter']) * controller.storage['prev_deriv_xy'][0]
    integral_x = controller.storage['integral_xy'][0] + error_x * dt
    integral_x = max(min(integral_x, PID_params['xy']['Imax']), -PID_params['xy']['Imax'])
    vx = PID_params['xy']['Kp'] * error_x + PID_params['xy']['Ki'] * integral_x + PID_params['xy']['Kd'] * filtered_deriv_x

    # Y轴同理
    raw_deriv_y = (error_y - controller.storage['prev_error_xy'][1]) / dt if dt > 1e-5 else 0
    filtered_deriv_y = PID_params['xy']['D_filter'] * raw_deriv_y + (1 - PID_params['xy']['D_filter']) * controller.storage['prev_deriv_xy'][1]
    integral_y = controller.storage['integral_xy'][1] + error_y * dt
    integral_y = max(min(integral_y, PID_params['xy']['Imax']), -PID_params['xy']['Imax'])
    vy = PID_params['xy']['Kp'] * error_y + PID_params['xy']['Ki'] * integral_y + PID_params['xy']['Kd'] * filtered_deriv_y

    # Z轴
    integral_z = controller.storage['integral_z'] + error_z * dt
    integral_z = max(min(integral_z, PID_params['z']['Imax']), -PID_params['z']['Imax'])
    derivative_z = (error_z - controller.storage['prev_error_z']) / dt if dt > 1e-5 else 0
    vz = PID_params['z']['Kp'] * error_z + PID_params['z']['Ki'] * integral_z + PID_params['z']['Kd'] * derivative_z

    # 偏航角（带微分滤波）
    raw_deriv_yaw = (yaw_error - controller.storage['prev_error_yaw']) / dt if dt > 1e-5 else 0
    filtered_deriv_yaw = PID_params['yaw']['Kd'] * raw_deriv_yaw + (1 - PID_params['yaw']['Kd']) * controller.storage['prev_deriv_yaw']
    integral_yaw = controller.storage['integral_yaw'] + yaw_error * dt
    integral_yaw = max(min(integral_yaw, PID_params['yaw']['Imax']), -PID_params['yaw']['Imax'])
    yaw_rate = PID_params['yaw']['Kp'] * yaw_error + PID_params['yaw']['Ki'] * integral_yaw + filtered_deriv_yaw

    # ====== 更新状态 ======
    controller.storage.update({
        'integral_xy': [integral_x, integral_y],
        'integral_z': integral_z,
        'integral_yaw': integral_yaw,
        'prev_error_xy': [error_x, error_y],
        'prev_error_z': error_z,
        'prev_error_yaw': yaw_error,
        'prev_deriv_xy': [filtered_deriv_x, filtered_deriv_y],
        'prev_deriv_yaw': filtered_deriv_yaw
    })

    # ====== 输出限幅优化 ======
    vx = max(min(vx, 2.0), -2.0)  # 放宽限幅以提高响应速度
    vy = max(min(vy, 2.0), -2.0)
    vz = max(min(vz, 1.0), -1.0)
    yaw_rate = max(min(yaw_rate, 2.0), -2.0)

    return (vx, vy, vz, yaw_rate)