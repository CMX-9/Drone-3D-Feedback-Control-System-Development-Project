def controller(state, target_pos, dt):
    # 状态格式: [x, y, z, roll, pitch, yaw] (单位：米，弧度)
    # 目标格式: (x, y, z, yaw) (单位：米，弧度)
    # dt: 时间步长 (秒)
    # 返回值: (vx, vy, vz, yaw_rate) (单位：米/秒，弧度/秒)

    # ====== PID参数配置 ======
    PID_params = {
        'xy': {'Kp': 0.8, 'Ki': 0.05, 'Kd': 0.3, 'Imax': 2.0},  # 水平面控制参数
        'z':  {'Kp': 1.2, 'Ki': 0.1,  'Kd': 0.5, 'Imax': 1.0},  # 高度控制参数
        'yaw':{'Kp': 0.6, 'Ki': 0.02, 'Kd': 0.2, 'Imax': 1.0}   # 偏航控制参数
    }

    # ====== 持久化状态初始化 ======
    if not hasattr(controller, "storage"):
        controller.storage = {
            # 轨迹跟踪相关
            'current_target': None,
            'trajectory': [],
            'step': 0,
            
            # PID状态存储
            'integral_xy': [0.0, 0.0],  # [x积分, y积分]
            'integral_z': 0.0,
            'integral_yaw': 0.0,
            'prev_error_xy': [0.0, 0.0],
            'prev_error_z': 0.0,
            'prev_error_yaw': 0.0
        }

    # ====== 轨迹管理模块 ======
    # 检测目标变化(位置容差0.1m，角度容差5度)
    target_changed = (
        controller.storage['current_target'] is None or
        any(abs(t - s) > 0.1 for t, s in zip(target_pos[:3], controller.storage['current_target'][:3])) or
        abs(target_pos[3] - controller.storage['current_target'][3]) > 0.087  # 5度=0.087rad
    )

    if target_changed:
        # 生成新轨迹时重置PID状态
        controller.storage.update({
            'integral_xy': [0.0, 0.0],
            'integral_z': 0.0,
            'integral_yaw': 0.0,
            'prev_error_xy': [0.0, 0.0],
            'prev_error_z': 0.0,
            'prev_error_yaw': 0.0
        })

        # 轨迹生成函数
        def generate_trajectory(start, end, dt, duration=3.0):
            """五次多项式轨迹生成"""
            num = max(1, int(round(duration/dt)))
            return [start + (end - start) * (10*(i/num)**3 -15*(i/num)**4 +6*(i/num)**5)
                    for i in range(num)]

        # 生成各轴轨迹
        x_traj = generate_trajectory(state[0], target_pos[0], dt)
        y_traj = generate_trajectory(state[1], target_pos[1], dt)
        z_traj = generate_trajectory(state[2], target_pos[2], dt)
        yaw_traj = generate_trajectory(state[5], target_pos[3], dt)

        # 轨迹对齐
        min_len = min(len(x_traj), len(y_traj), len(z_traj), len(yaw_traj))
        controller.storage['trajectory'] = list(zip(
            x_traj[:min_len],
            y_traj[:min_len],
            z_traj[:min_len],
            yaw_traj[:min_len]
        ))
        controller.storage['step'] = 0
        controller.storage['current_target'] = target_pos

    # ====== 获取当前目标点 ======
    if controller.storage['step'] < len(controller.storage['trajectory']):
        target = controller.storage['trajectory'][controller.storage['step']]
        controller.storage['step'] += 1
    else:
        target = target_pos  # 保持最终目标

    # ====== 坐标系转换 ======
    current_yaw = state[5]
    dx = target[0] - state[0]
    dy = target[1] - state[1]
    
    # 泰勒展开近似三角函数
    cos_y = 1.0 - current_yaw**2/2 + current_yaw**4/24  # cos近似
    sin_y = current_yaw - current_yaw**3/6 + current_yaw**5/120  # sin近似
    
    # 机体坐标系误差
    error_x = dx * cos_y + dy * sin_y
    error_y = -dx * sin_y + dy * cos_y
    error_z = target[2] - state[2]
    
    # 偏航角误差处理
    yaw_error = target[3] - current_yaw
    yaw_error = (yaw_error + 3.1415926535) % 6.283185307 - 3.1415926535  # 角度包裹

    # ====== PID计算核心 ======
    # X轴PID
    integral_x = controller.storage['integral_xy'][0] + error_x * dt
    integral_x = max(min(integral_x, PID_params['xy']['Imax']), -PID_params['xy']['Imax'])
    derivative_x = (error_x - controller.storage['prev_error_xy'][0]) / dt if dt > 1e-5 else 0
    vx = (PID_params['xy']['Kp'] * error_x +
          PID_params['xy']['Ki'] * integral_x +
          PID_params['xy']['Kd'] * derivative_x)

    # Y轴PID
    integral_y = controller.storage['integral_xy'][1] + error_y * dt
    integral_y = max(min(integral_y, PID_params['xy']['Imax']), -PID_params['xy']['Imax'])
    derivative_y = (error_y - controller.storage['prev_error_xy'][1]) / dt if dt > 1e-5 else 0
    vy = (PID_params['xy']['Kp'] * error_y +
          PID_params['xy']['Ki'] * integral_y +
          PID_params['xy']['Kd'] * derivative_y)

    # Z轴PID
    integral_z = controller.storage['integral_z'] + error_z * dt
    integral_z = max(min(integral_z, PID_params['z']['Imax']), -PID_params['z']['Imax'])
    derivative_z = (error_z - controller.storage['prev_error_z']) / dt if dt > 1e-5 else 0
    vz = (PID_params['z']['Kp'] * error_z +
          PID_params['z']['Ki'] * integral_z +
          PID_params['z']['Kd'] * derivative_z)

    # 偏航角PID
    integral_yaw = controller.storage['integral_yaw'] + yaw_error * dt
    integral_yaw = max(min(integral_yaw, PID_params['yaw']['Imax']), -PID_params['yaw']['Imax'])
    derivative_yaw = (yaw_error - controller.storage['prev_error_yaw']) / dt if dt > 1e-5 else 0
    yaw_rate = (PID_params['yaw']['Kp'] * yaw_error +
                PID_params['yaw']['Ki'] * integral_yaw +
                PID_params['yaw']['Kd'] * derivative_yaw)

    # ====== 更新存储状态 ======
    controller.storage.update({
        'integral_xy': [integral_x, integral_y],
        'integral_z': integral_z,
        'integral_yaw': integral_yaw,
        'prev_error_xy': [error_x, error_y],
        'prev_error_z': error_z,
        'prev_error_yaw': yaw_error
    })

    # ====== 输出限幅 ======
    vx = max(min(vx, 1.0), -1.0)
    vy = max(min(vy, 1.0), -1.0)
    vz = max(min(vz, 0.5), -0.5)
    yaw_rate = max(min(yaw_rate, 1.0), -1.0)

    return (vx, vy, vz, yaw_rate)