import numpy as np

def controller(state, target_pos, dt):
    """完整PID控制器 - 包含滤波修复和震荡抑制"""
    
    # ================= 配置参数 =================
    MAX_HORIZ_VEL = 1.2           # 水平最大速度(m/s)
    MAX_VERT_VEL = 0.8            # 垂直最大速度(m/s)
    MAX_YAW_RATE = 1.0            # 偏航角速度(rad/s)
    OSCILLATION_THRESHOLD = 0.12   # 震荡检测阈值(m)

    PID_params = {
        # 位置环参数
        'pos_xy': {'Kp': 0.3, 'Ki': 0.005, 'Kd': 0.4, 'Imax': 0.3, 'deadzone': 0.05},
        'pos_z':  {'Kp': 0.5, 'Ki': 0.01,  'Kd': 0.35, 'Imax': 0.2},
        'pos_yaw':{'Kp': 0.25,'Ki': 0.001, 'Kd': 0.15, 'Imax': 0.1},
        
        # 速度环参数
        'vel_xy': {'Kp': 0.7, 'Ki': 0.03,  'Kd': 0.25, 'Imax': 0.25},
        'vel_z':  {'Kp': 0.9, 'Ki': 0.04,  'Kd': 0.2,  'Imax': 0.15},
        
        # 滤波器参数
        'D_filter': 0.4,    # 微分项滤波系数
        'vel_filter': 0.35  # 速度滤波系数
    }

    # ============== 持久化状态初始化 ==============
    if not hasattr(controller, "storage"):
        controller.storage = {
            # 外环状态
            'pos_integral_xy': [0.0, 0.0],
            'pos_integral_z': 0.0,
            'pos_integral_yaw': 0.0,
            'pos_prev_error_xy': [0.0, 0.0],
            
            # 速度滤波状态（独立存储各轴）
            'filtered_vel_x': 0.0,
            'filtered_vel_y': 0.0,
            'filtered_vel_z': 0.0,
            
            # 震荡检测
            'oscillation_counter': 0,
            'last_error_sign': [1, 1],
            
            # 目标状态
            'current_target': None,
            'last_target': None
        }

    # ============== 核心函数 ==============
    def low_pass_filter(new_val, prev_val, alpha):
        """标量低通滤波器"""
        return alpha * new_val + (1 - alpha) * prev_val

    def global_to_body(error_global, yaw):
        """全局坐标系到机体坐标系转换"""
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        return (
            error_global[0] * cos_y + error_global[1] * sin_y,
            -error_global[0] * sin_y + error_global[1] * cos_y
        )

    def detect_oscillation(error_x, error_y):
        """改进的震荡检测逻辑"""
        current_sign_x = 1 if error_x >= 0 else -1
        current_sign_y = 1 if error_y >= 0 else -1
        
        # X轴符号变化检测
        if current_sign_x != controller.storage['last_error_sign'][0] and abs(error_x) > OSCILLATION_THRESHOLD:
            controller.storage['oscillation_counter'] += 1
        # Y轴符号变化检测
        elif current_sign_y != controller.storage['last_error_sign'][1] and abs(error_y) > OSCILLATION_THRESHOLD:
            controller.storage['oscillation_counter'] += 1
        else:
            controller.storage['oscillation_counter'] = max(0, controller.storage['oscillation_counter'] - 1)
        
        controller.storage['last_error_sign'] = [current_sign_x, current_sign_y]
        return controller.storage['oscillation_counter'] > 3

    class CascadePID:
        """级联PID控制器（修复滤波版本）"""
        def __init__(self, axis_type):
            self.axis_type = axis_type  # 'x'/'y'/'z'
            self.pos_integral = 0.0
            self.pos_prev_error = 0.0
            self.pos_prev_deriv = 0.0
            self.vel_integral = 0.0
            self.vel_prev_error = 0.0
            
        def update(self, error, raw_velocity, dt):
            # 选择参数集
            pos_params = PID_params['pos_xy'] if self.axis_type in ['x','y'] else PID_params['pos_z']
            vel_params = PID_params['vel_xy'] if self.axis_type in ['x','y'] else PID_params['vel_z']
            
            # ==== 死区处理 ====
            if abs(error) < pos_params.get('deadzone', 0.0):
                return 0.0
                
            # ==== 位置环计算 ====
            # 积分项限幅
            self.pos_integral = np.clip(
                self.pos_integral + error * dt,
                -pos_params['Imax'],
                pos_params['Imax']
            )
            
            # 微分项计算（带滤波）
            deriv = (error - self.pos_prev_error) / max(dt, 1e-5)
            deriv = PID_params['D_filter'] * deriv + (1 - PID_params['D_filter']) * self.pos_prev_deriv
            
            # 目标速度计算
            target_vel = pos_params['Kp']*error + pos_params['Ki']*self.pos_integral + pos_params['Kd']*deriv
            
            # ==== 速度环计算 ====
            # 获取滤波后的速度
            if self.axis_type == 'x':
                filtered_vel = controller.storage['filtered_vel_x']
            elif self.axis_type == 'y':
                filtered_vel = controller.storage['filtered_vel_y']
            else:
                filtered_vel = controller.storage['filtered_vel_z']
            
            vel_error = target_vel - filtered_vel
            
            # 速度积分项
            self.vel_integral = np.clip(
                self.vel_integral + vel_error * dt,
                -vel_params['Imax'],
                vel_params['Imax']
            )
            
            # 速度微分
            vel_deriv = (vel_error - self.vel_prev_error) / max(dt, 1e-5)
            
            # 最终输出
            output = vel_params['Kp']*vel_error + vel_params['Ki']*self.vel_integral + vel_params['Kd']*vel_deriv
            
            # ==== 状态更新 ====
            self.pos_prev_error = error
            self.pos_prev_deriv = deriv
            self.vel_prev_error = vel_error
            
            return output

    # ============== 主控制流程 ==============
    current_pos = state[:3]
    current_vel = state[3:6]
    current_yaw = state[5] if len(state) > 5 else 0.0

    # 初始化目标位置
    if controller.storage['current_target'] is None:
        controller.storage['current_target'] = np.array(target_pos[:3])
        return (0.0, 0.0, 0.0, 0.0)  # 初始返回零指令

    # 更新速度滤波
    controller.storage['filtered_vel_x'] = low_pass_filter(
        current_vel[0],
        controller.storage['filtered_vel_x'],
        PID_params['vel_filter']
    )
    controller.storage['filtered_vel_y'] = low_pass_filter(
        current_vel[1],
        controller.storage['filtered_vel_y'],
        PID_params['vel_filter']
    )
    controller.storage['filtered_vel_z'] = low_pass_filter(
        current_vel[2],
        controller.storage['filtered_vel_z'],
        PID_params['vel_filter']
    )

    # 坐标系转换
    error_global = [
        controller.storage['current_target'][0] - current_pos[0],
        controller.storage['current_target'][1] - current_pos[1]
    ]
    error_body = global_to_body(error_global, current_yaw)
    error_z = controller.storage['current_target'][2] - current_pos[2]

    # 创建控制器实例
    pid_x = CascadePID('x')
    pid_y = CascadePID('y')
    pid_z = CascadePID('z')

    # 执行控制计算
    vx = pid_x.update(error_body[0], current_vel[0], dt)
    vy = pid_y.update(error_body[1], current_vel[1], dt)
    vz = pid_z.update(error_z, current_vel[2], dt)

    # 震荡检测与处理
    if detect_oscillation(error_body[0], error_body[1]):
        vx *= 0.6
        vy *= 0.6
        MAX_HORIZ_VEL = 0.8  # 临时降低速度限制

    # 偏航控制
    yaw_error = (target_pos[3] - current_yaw + np.pi) % (2*np.pi) - np.pi
    controller.storage['pos_integral_yaw'] = np.clip(
        controller.storage['pos_integral_yaw'] + yaw_error * dt,
        -PID_params['pos_yaw']['Imax'],
        PID_params['pos_yaw']['Imax']
    )
    yaw_deriv = (yaw_error - controller.storage.get('pos_prev_error_yaw', 0.0)) / max(dt, 1e-5)
    yaw_rate = PID_params['pos_yaw']['Kp']*yaw_error + PID_params['pos_yaw']['Ki']*controller.storage['pos_integral_yaw'] + PID_params['pos_yaw']['Kd']*yaw_deriv
    controller.storage['pos_prev_error_yaw'] = yaw_error

    # 最终输出限幅
    return (
        np.clip(vx, -MAX_HORIZ_VEL, MAX_HORIZ_VEL),
        np.clip(vy, -MAX_HORIZ_VEL, MAX_HORIZ_VEL),
        np.clip(vz, -MAX_VERT_VEL, MAX_VERT_VEL),
        np.clip(yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE)
    )