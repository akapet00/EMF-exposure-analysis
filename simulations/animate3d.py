import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# data
filename = 'deltaT_Nx31Ny31Nz31_t860_a4'
sim_time = 860
t = np.linspace(0, sim_time, 50)
N = [31] * 3
area = (0.02, 0.02)
x = np.linspace(-area[0]/2, area[0]/2, N[0])
y = np.linspace(-area[0]/2, area[1]/2, N[1])
xmesh, ymesh = np.meshgrid(x, y)
T = np.load(f'{filename}.npy', allow_pickle=True, fix_imports=True)

# init and config fig
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim3d(x.min()*1.05, x.max()*1.05)
ax.set_ylim3d(y.min()*1.05, y.max()*1.05)
ax.set_zlim3d(T[-1, :, :, 0].min(), T[-1, :, :, 0].max())
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.zaxis.labelpad = 10
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_zlabel('$\\Delta T$ [°C]')
ax.w_xaxis.set_pane_color((0, 0, 0))
ax.w_yaxis.set_pane_color((0, 0, 0))
ax.w_zaxis.set_pane_color((0, 0, 0))

# animate
def init():
    ax.plot_surface(xmesh, ymesh, T[0, :, :, 0], cmap='RdYlBu_r', antialiased=True)

def animate(frame):
    ax.plot_surface(xmesh, ymesh, T[frame, :, :, 0], cmap='RdYlBu_r', antialiased=True)
    ax.set_title(f'$t$ = {int(t[frame])} s; $\\Delta T_{{max}}^{{surf}}$ = {T[frame, :, :, 0].max():.3f} °C')

anim = FuncAnimation(fig, animate, init_func=init, frames=np.arange(1, t.size), interval=1)
anim.save(f'{filename}.gif', writer='imagemagick', fps=20)