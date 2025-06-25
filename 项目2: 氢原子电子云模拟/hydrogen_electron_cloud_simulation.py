import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

# Physical constants
a = 5.29e-2  # Bohr radius (nm)
r0 = 0.25     # Convergence radius (nm)

# Hydrogen atom ground state electron probability density function
def probability_density(r):
    return (4 * r**2 / a**3) * np.exp(-2 * r / a)

# Calculate maximum probability density for given a
def calculate_D_max(a_val):
    # Find the maximum by solving dD/dr = 0
    # Maximum occurs at r = a
    return probability_density(a_val)

# Generate electron positions following the probability density distribution
def generate_electron_positions(num_points, a_val, r0_val):
    D_max = calculate_D_max(a_val)
    positions = []
    
    while len(positions) < num_points:
        # Generate random point in spherical coordinates
        r_candidate = np.random.uniform(0, r0_val)
        theta = np.arccos(2 * np.random.random() - 1)  # Polar angle [0, π]
        phi = 2 * np.pi * np.random.random()           # Azimuthal angle [0, 2π]
        
        # Acceptance-rejection sampling
        if np.random.random() <= probability_density(r_candidate) / D_max:
            # Convert to Cartesian coordinates
            x = r_candidate * np.sin(theta) * np.cos(phi)
            y = r_candidate * np.sin(theta) * np.sin(phi)
            z = r_candidate * np.cos(theta)
            positions.append((x, y, z))
    
    return np.array(positions)

# Fit function for radial distribution
def fit_function(r, A, B):
    return A * r**2 * np.exp(-B * r)

# Visualize electron cloud
def visualize_electron_cloud(positions, a_val, r0_val):
    fig = plt.figure(figsize=(12, 10))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(positions[:,0], positions[:,1], positions[:,2], 
                s=0.5, alpha=0.3, c='b')
    ax1.set_title('3D Electron Cloud Distribution')
    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Y (nm)')
    ax1.set_zlabel('Z (nm)')
    
    # XY plane projection
    ax2 = fig.add_subplot(222)
    ax2.scatter(positions[:,0], positions[:,1], s=0.5, alpha=0.3)
    ax2.set_title('XY Plane Projection')
    ax2.set_xlabel('X (nm)')
    ax2.set_ylabel('Y (nm)')
    ax2.set_aspect('equal')
    
    # Radial distribution analysis
    radii = np.linalg.norm(positions, axis=1)
    hist, bins = np.histogram(radii, bins=100, density=True)
    r_vals = (bins[:-1] + bins[1:]) / 2
    
    # Improved fitting
    try:
        # Initial parameter guesses (A, B)
        p0 = [4/(a_val**3), 2/a_val]
        # Fit the curve
        params, cov = curve_fit(fit_function, r_vals, hist, p0=p0, maxfev=5000)
        
        # Generate fitted curve with more points for smoothness
        r_fit = np.linspace(0, r0_val, 200)
        fit_curve = fit_function(r_fit, *params)
        
        fit_label = f'Fit: A={params[0]:.2f}, B={params[1]:.2f}'
    except RuntimeError:
        print("Curve fitting failed. Using theoretical curve instead.")
        r_fit = np.linspace(0, r0_val, 200)
        fit_curve = probability_density(r_fit)
        fit_label = 'Theoretical Distribution'
    
    ax3 = fig.add_subplot(212)
    ax3.plot(r_vals, hist, 'b-', label='Simulated Distribution')
    ax3.plot(r_fit, fit_curve, 'r-', linewidth=2, label=fit_label)
    ax3.set_title('Radial Probability Density Distribution')
    ax3.set_xlabel('Radius r (nm)')
    ax3.set_ylabel('Probability Density')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

# Parameter effect analysis
def analyze_parameter_effects():
    # Test different parameter values
    a_values = [0.04, 0.0529, 0.07]
    r0_values = [0.2, 0.25, 0.3]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Parameter Effects on Electron Cloud Distribution', fontsize=16)
    
    # Analyze effect of Bohr radius a
    for i, a_val in enumerate(a_values):
        # Generate electron positions
        positions = generate_electron_positions(20000, a_val, r0)
        radii = np.linalg.norm(positions, axis=1)
        
        # Plot radial distribution
        hist, bins = np.histogram(radii, bins=100, density=True)
        r_vals = (bins[:-1] + bins[1:]) / 2
        axes[0, i].plot(r_vals, hist, label=f'Simulated (a = {a_val} nm)')
        
        # Theoretical curve
        r_theory = np.linspace(0, r0, 200)
        theory_curve = probability_density(r_theory)
        axes[0, i].plot(r_theory, theory_curve, 'r--', label='Theoretical')
        
        axes[0, i].set_title(f'Bohr Radius a = {a_val} nm')
        axes[0, i].set_xlabel('Radius r (nm)')
        axes[0, i].set_ylabel('Probability Density')
        axes[0, i].grid(True)
        axes[0, i].legend()
    
    # Analyze effect of convergence radius r0
    for i, r0_val in enumerate(r0_values):
        # Generate electron positions
        positions = generate_electron_positions(20000, a, r0_val)
        radii = np.linalg.norm(positions, axis=1)
        
        # Plot radial distribution
        hist, bins = np.histogram(radii, bins=100, density=True)
        r_vals = (bins[:-1] + bins[1:]) / 2
        axes[1, i].plot(r_vals, hist, label=f'Simulated (r0 = {r0_val} nm)')
        
        # Theoretical curve
        r_theory = np.linspace(0, r0_val, 200)
        theory_curve = probability_density(r_theory)
        axes[1, i].plot(r_theory, theory_curve, 'r--', label='Theoretical')
        
        axes[1, i].set_title(f'Convergence Radius r0 = {r0_val} nm')
        axes[1, i].set_xlabel('Radius r (nm)')
        axes[1, i].set_ylabel('Probability Density')
        axes[1, i].grid(True)
        axes[1, i].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Main program
if __name__ == "__main__":
    # Generate electron positions
    electron_positions = generate_electron_positions(50000, a, r0)
    
    # Visualize electron cloud
    visualize_electron_cloud(electron_positions, a, r0)
    
    # Analyze parameter effects
    analyze_parameter_effects()
