import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt
import numpy as np

def plot_3d_pbe_solution(data,PBE, t_step, t_steps, x_min= 1e-6, x_max=8e-3):
    # Create mask for data within limits
    def mask_data_by_limits(L, t, data, x_min, x_max):
        """Mask data outside the specified L limits"""
        mask = (L_grid >= x_min) & (L_grid <= x_max)

        # Create masked arrays
        L_masked = np.ma.array(L_grid, mask=~mask)
        t_masked = np.ma.array(t_grid, mask=~mask)
        data_masked = np.ma.array(data, mask=~mask)

        return L_masked, t_masked, data_masked


    # Using your data and coordinates
    L = PBE.L_i
    t = np.linspace(0, t_step * t_steps, data.shape[0])
    L_grid, t_grid = np.meshgrid(L, t)



    # Mask the data
    L_masked, t_masked, data_masked = mask_data_by_limits(L_grid, t_grid, data, x_min, x_max)

    # Create the figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot with masked data
    surf = ax.plot_surface(
        L_masked, t_masked, data_masked,
        cmap='viridis',
        edgecolor='none',
        alpha=1.0,
        antialiased=True
    )

    # Set axis labels
    ax.set_xlabel('L')
    ax.set_ylabel('Time t')
    ax.set_zlabel('N(L)')

    # Set x-axis to log scale and limits
    # ax.set_xscale('log')
    ax.set_xlim(x_min, x_max)

    # Set the view angle
    ax.view_init(elev=20, azim=45)

    # Make background panes white
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')

    # # Add colorbar
    # cbar = fig.colorbar(surf, shrink=0.8, aspect=20)

    # Adjust layout
    plt.tight_layout()

    # Make it interactive
    plt.ion()

    # Show the plot
    plt.show(block=True)