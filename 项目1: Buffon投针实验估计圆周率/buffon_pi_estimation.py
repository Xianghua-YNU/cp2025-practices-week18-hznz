import numpy as np
import matplotlib.pyplot as plt
import time

def buffon_needle_experiment(num_trials, needle_length=1, line_spacing=2):
    """
    执行Buffon投针实验
    
    参数:
    num_trials: 实验次数
    needle_length: 针的长度 (默认1)
    line_spacing: 平行线的间距 (默认2)
    
    返回:
    pi_estimate: π的估计值
    intersection_count: 针与线相交的次数
    """
    # 记录开始时间
    start_time = time.time()
    
    # 生成随机数据
    # 针中心到最近线的距离 (0到line_spacing/2之间)
    y = np.random.uniform(0, line_spacing/2, num_trials)
    # 针的角度 (0到π/2之间)
    theta = np.random.uniform(0, np.pi/2, num_trials)
    
    # 计算针是否与线相交
    # 相交条件: y <= (needle_length/2) * sin(theta)
    intersections = y <= (needle_length/2) * np.sin(theta)
    intersection_count = np.sum(intersections)
    
    # 计算π的估计值
    if intersection_count > 0:
        pi_estimate = num_trials / intersection_count
    else:
        pi_estimate = 0  # 避免除零错误
    
    # 计算执行时间
    execution_time = time.time() - start_time
    
    return pi_estimate, intersection_count, execution_time

# 设置不同的实验次数
trial_counts = [100, 1000, 10000, 100000, 1000000, 5000000]

# 存储实验结果
results = []
actual_pi = np.pi

# 执行实验
for n in trial_counts:
    pi_est, count, time_taken = buffon_needle_experiment(n)
    error = abs(pi_est - actual_pi) if pi_est > 0 else actual_pi
    relative_error = (error / actual_pi) * 100  # 百分比误差
    
    results.append({
        'trials': n,
        'pi_estimate': pi_est,
        'intersections': count,
        'time': time_taken,
        'error': error,
        'relative_error': relative_error
    })
    
    print(f"实验次数: {n:>8,} | π估计值: {pi_est:.6f} | 相交次数: {count} | "
          f"误差: {error:.6f} | 相对误差: {relative_error:.2f}% | "
          f"耗时: {time_taken:.4f}秒")

# 可视化结果
plt.figure(figsize=(14, 10))

# 1. π估计值随实验次数变化
plt.subplot(2, 2, 1)
pi_estimates = [r['pi_estimate'] for r in results]
plt.plot(trial_counts, pi_estimates, 'o-', label='π估计值')
plt.axhline(y=actual_pi, color='r', linestyle='--', label='真实π值')
plt.xscale('log')
plt.xlabel('实验次数 (对数尺度)')
plt.ylabel('π估计值')
plt.title('π估计值随实验次数变化')
plt.legend()
plt.grid(True, which="both", ls="--")

# 2. 相对误差随实验次数变化
plt.subplot(2, 2, 2)
relative_errors = [r['relative_error'] for r in results]
plt.plot(trial_counts, relative_errors, 's-', color='green')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('实验次数 (对数尺度)')
plt.ylabel('相对误差 (%)')
plt.title('相对误差随实验次数变化 (对数尺度)')
plt.grid(True, which="both", ls="--")

# 3. 相交比例随实验次数变化
plt.subplot(2, 2, 3)
intersection_ratios = [r['intersections'] / r['trials'] for r in results]
expected_ratio = 2 / (actual_pi * 2)  # P = (2l)/(πd), 其中l=1, d=2
plt.plot(trial_counts, intersection_ratios, 'd-', color='purple')
plt.axhline(y=expected_ratio, color='r', linestyle='--', label='理论相交概率')
plt.xscale('log')
plt.xlabel('实验次数 (对数尺度)')
plt.ylabel('相交比例')
plt.title('相交比例随实验次数变化')
plt.legend()
plt.grid(True, which="both", ls="--")

# 4. 执行时间随实验次数变化
plt.subplot(2, 2, 4)
execution_times = [r['time'] for r in results]
plt.plot(trial_counts, execution_times, '*-', color='orange')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('实验次数 (对数尺度)')
plt.ylabel('执行时间 (秒)')
plt.title('执行时间随实验次数变化 (对数尺度)')
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig('buffon_results.png', dpi=300)
plt.show()

# 输出表格形式的结果
print("\n实验结果总结:")
print("实验次数   | π估计值    | 相交次数   | 误差       | 相对误差(%) | 耗时(秒)")
print("-----------------------------------------------------------------------")
for r in results:
    print(f"{r['trials']:9,} | {r['pi_estimate']:8.6f} | {r['intersections']:9,} | "
          f"{r['error']:8.6f} | {r['relative_error']:7.4f}% | {r['time']:.6f}")
