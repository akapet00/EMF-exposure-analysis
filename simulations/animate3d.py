import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# data
filename = 'deltaT_source-HWdipole_grid-101x101x21_surface-4cm_timegrid-51_simtime-360s'
sim_time = 360
tg = 51
t = np.linspace(0, sim_time, tg)
N = [101, 101, 21]
area = (0.02, 0.02)
x = np.linspace(-area[0]/2, area[0]/2, N[0])
y = np.linspace(-area[0]/2, area[1]/2, N[1])
xmesh, ymesh = np.meshgrid(x, y)
deltaT = np.load(f'{filename}.npy', allow_pickle=True, fix_imports=True)

# init and config fig
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim3d(x.min() * 1.05, x.max() * 1.05)
ax.set_ylim3d(y.min() * 1.05, y.max() * 1.05)
ax.set_zlim3d(deltaT[-1, :, :, 0].min(), deltaT[-1, :, :, 0].max())
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.zaxis.labelpad = 10
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_zlabel('$\\Delta T$ [°C]')
ax.w_xaxis.set_pane_color((0, 0, 0))
ax.w_yaxis.set_pane_color((0, 0, 0))
ax.w_zaxis.set_pane_color((0, 0, 0))


def init():
    ax.plot_surface(xmesh, ymesh, deltaT[0, :, :, 0], cmap='viridis',
                    antialiased=True)


def animate(frame):
    ax.plot_trisurf(xmesh.ravel(), ymesh.ravel(),
                    deltaT[frame, :, :, 0].ravel(), cmap='viridis',
                    antialiased=False)
    ax.set_title(f'$t = {int(t[frame])}$ s; '
                 f'$\\Delta T_{{max}} = {deltaT[frame, :, :, 0].max():.3f}$ °C')


anim = FuncAnimation(fig, animate, init_func=init,
                     frames=np.arange(1, t.size), interval=1)
anim.save(f'{filename}.gif', fps=20)
