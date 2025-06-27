import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from tqdm import tqdm

class ChainReactionSimulator:
    def __init__(self, size=50, p_fission=0.05, p_absorb=0.02, n_initial=5, max_steps=200):
        """
        初始化链式反应模拟器
        
        参数:
        size: 反应区域大小 (size x size)
        p_fission: 裂变概率 (产生新中子)
        p_absorb: 吸收概率 (中子被吸收)
        n_initial: 初始中子数
        max_steps: 最大模拟步数
        """
        self.size = size
        self.p_fission = p_fission
        self.p_absorb = p_absorb
        self.n_initial = n_initial
        self.max_steps = max_steps
        
        # 初始化网格 (0=空, 1=有中子)
        self.grid = np.zeros((size, size), dtype=int)
        
        # 随机放置初始中子
        positions = np.random.choice(size*size, n_initial, replace=False)
        for pos in positions:
            x, y = pos // size, pos % size
            self.grid[x, y] = 1
    
    def step(self):
        """执行一步模拟"""
        new_grid = np.zeros_like(self.grid)
        neutron_count = 0
        
        # 遍历所有格子
        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x, y] == 1:  # 当前位置有中子
                    # 中子可能被吸收
                    if np.random.random() < self.p_absorb:
                        continue
                    
                    # 中子未被吸收，可能裂变
                    if np.random.random() < self.p_fission:
                        # 产生2个新中子 (随机方向移动)
                        for _ in range(2):
                            dx, dy = np.random.randint(-1, 2, 2)
                            new_x, new_y = (x + dx) % self.size, (y + dy) % self.size
                            new_grid[new_x, new_y] = 1
                            neutron_count += 1
                    else:
                        # 不裂变，中子移动
                        dx, dy = np.random.randint(-1, 2, 2)
                        new_x, new_y = (x + dx) % self.size, (y + dy) % self.size
                        new_grid[new_x, new_y] = 1
                        neutron_count += 1
        
        self.grid = new_grid
        return neutron_count
    
    def simulate(self, visualize=False):
        """执行完整模拟"""
        neutron_counts = []
        
        if visualize:
            plt.figure(figsize=(10, 8))
            im = plt.imshow(self.grid, cmap='hot', interpolation='nearest')
            plt.title(f"Step 0, Neutrons: {self.n_initial}")
            plt.colorbar()
            
            def update(frame):
                count = self.step()
                neutron_counts.append(count)
                im.set_array(self.grid)
                plt.title(f"Step {frame+1}, Neutrons: {count}")
                return [im]
            
            ani = FuncAnimation(plt.gcf(), update, frames=self.max_steps, 
                               interval=100, blit=True, repeat=False)
            plt.show()
        else:
            for _ in range(self.max_steps):
                count = self.step()
                neutron_counts.append(count)
        
        return neutron_counts
    
    def multiple_simulations(self, n_simulations=100):
        """执行多次模拟并收集统计数据"""
        all_counts = []
        critical_count = 0
        
        for _ in tqdm(range(n_simulations), desc="Running simulations"):
            # 重置模拟器
            self.grid = np.zeros((self.size, self.size), dtype=int)
            positions = np.random.choice(self.size*self.size, self.n_initial, replace=False)
            for pos in positions:
                x, y = pos // self.size, pos % self.size
                self.grid[x, y] = 1
            
            # 执行模拟
            counts = self.simulate(visualize=False)
            all_counts.append(counts)
            
            # 判断是否达到临界状态 (最后10步平均中子数大于初始值)
            if np.mean(counts[-10:]) > self.n_initial:
                critical_count += 1
        
        return all_counts, critical_count / n_simulations

def parameter_study():
    """参数影响分析"""
    sizes = [30, 50, 70]
    p_fission_values = np.linspace(0.01, 0.1, 10)
    p_absorb_values = np.linspace(0.01, 0.1, 10)
    
    # 研究裂变概率影响 (固定吸收概率)
    critical_probs_fission = []
    for p_fission in p_fission_values:
        sim = ChainReactionSimulator(p_fission=p_fission, p_absorb=0.03)
        _, crit_prob = sim.multiple_simulations(50)
        critical_probs_fission.append(crit_prob)
    
    # 研究吸收概率影响 (固定裂变概率)
    critical_probs_absorb = []
    for p_absorb in p_absorb_values:
        sim = ChainReactionSimulator(p_fission=0.05, p_absorb=p_absorb)
        _, crit_prob = sim.multiple_simulations(50)
        critical_probs_absorb.append(crit_prob)
    
    # 绘制结果
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(p_fission_values, critical_probs_fission, 'o-')
    plt.xlabel('Fission Probability')
    plt.ylabel('Critical Probability')
    plt.title('Effect of Fission Probability (p_absorb=0.03)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(p_absorb_values, critical_probs_absorb, 'o-')
    plt.xlabel('Absorption Probability')
    plt.ylabel('Critical Probability')
    plt.title('Effect of Absorption Probability (p_fission=0.05)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_single_simulation():
    """可视化单次模拟"""
    sim = ChainReactionSimulator(size=50, p_fission=0.06, p_absorb=0.03, n_initial=10)
    neutron_counts = sim.simulate(visualize=True)
    
    # 绘制中子数随时间变化
    plt.figure(figsize=(10, 5))
    plt.plot(neutron_counts)
    plt.xlabel('Time Step')
    plt.ylabel('Number of Neutrons')
    plt.title('Neutron Population Over Time')
    plt.grid(True)
    plt.show()

# 运行示例
if __name__ == "__main__":
    print("1. Visualizing a single simulation...")
    visualize_single_simulation()
    
    print("\n2. Running parameter study...")
    parameter_study()
