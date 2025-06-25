import numpy as np
from time import time

def generate_samples(N):
    """生成服从p(x) = 1/(2√x)分布的随机数"""
    U = np.random.random(N)
    X = U * U
    return X

def estimate_integral(N):
    """估计积分值"""
    X = generate_samples(N)
    f = 1 / (np.exp(X) + 1)
    I_estimate = 2 * np.mean(f)
    return I_estimate, f

def estimate_error(f, N):
    """估计统计误差"""
    mean_f = np.mean(f)
    mean_f2 = np.mean(f * f)
    var_f = mean_f2 - mean_f * mean_f
    sigma = np.sqrt(var_f / N)
    return sigma

def main():
    # 设置随机数种子
    np.random.seed(42)
    
    # 主要计算
    N = 1000000
    start_time = time()
    I_estimate, f = estimate_integral(N)
    sigma = estimate_error(f, N)
    end_time = time()
    
    # 输出结果
    print("\n=== 积分计算结果 ===")
    print(f"采样点数量: {N:,}")
    print(f"积分估计值: {I_estimate:.6f} ± {sigma:.6f}")
    print(f"计算时间: {end_time - start_time:.2f} 秒")
    
    # 进行多次实验
    print("\n=== 稳定性验证 ===")
    n_experiments = 5
    results = []
    for i in range(n_experiments):
        I_est, f = estimate_integral(N)
        sig = estimate_error(f, N)
        results.append(I_est)
        print(f"实验 {i+1}: {I_est:.6f} ± {sig:.6f}")
    
    print(f"\n平均值: {np.mean(results):.6f}")
    print(f"标准差: {np.std(results):.6f}")

if __name__ == "__main__":
    main()
