# import asyncio
import time
import tkinter as tk
from scipy.integrate import solve_ivp
import numpy as np


def ovalcoords(xy):
    ret = np.empty(4)
    ret[:2] = xy - 10.0
    ret[2:] = xy + 10.0
    return ret

def rotpts2d(points, angle, center):
    theta = (angle / 180.0) * np.pi
    cos = np.cos(theta)
    sin = np.sin(theta)
    rm = np.array(([[cos, -sin],[sin, cos]]))
    cpoints = points - center
    ret = np.matmul(cpoints, rm)
    ret += center
    return ret


def dsdt(t, s):
    G = 5.0
    x0, y0, x1, y1, x2, y2, vx0, vy0, vx1, vy1, vx2, vy2 = s
    k01 = 1 / ((x1 - x0)**2 + (y1 - y0)**2) ** (1.5)
    k12 = 1 / ((x2 - x1)**2 + (y2 - y1)**2) ** (1.5)
    k02 = 1 / ((x2 - x0)**2 + (y2 - y0)**2) ** (1.5)
    dvx0dt = G*(k01 * (x1 - x0) + k02 * (x2 - x0))
    dvy0dt = G*(k01 * (y1 - y0) + k02 * (y2 - y0))
    dvx1dt = G*(k01 * (x0 - x1) + k12 * (x2 - x1))
    dvy1dt = G*(k01 * (y0 - y1) + k12 * (y2 - y1))
    dvx2dt = G*(k02 * (x0 - x2) + k12 * (x1 - x2))
    dvy2dt = G*(k02 * (y0 - y2) + k12 * (y1 - y2))
    return np.array([vx0, vy0, vx1, vy1, vx2, vy2, dvx0dt, dvy0dt, dvx1dt, dvy1dt, dvx2dt, dvy2dt])

def get_boundary_events():
    ret = []
    bounds = [10.0, 10.0, 750.0, 750.0]
    for i in range(3):
        for j in range(4):
            ret.append(
                lambda t, y, i=i, j=j: (-1)**(j//2) * (y[2*i+j%2] - bounds[j]) if t != 0.0 else 1e-3
            )
    for func in ret:
        func.terminal=True
    return ret

def get_collision_events():
    e01 = lambda t, y: np.linalg.norm(y[:2] - y[2:4]) - 20.0 if t != 0 else 1e-3
    e02 = lambda t, y: np.linalg.norm(y[:2] - y[4:6]) - 20.0 if t != 0 else 1e-3
    e12 = lambda t, y: np.linalg.norm(y[2:4] - y[4:6]) - 20.0 if t != 0 else 1e-3
    e01.terminal = True
    e02.terminal = True
    e12.terminal = True
    return [e01, e02, e12]

def new_vs(xys, vs):
    xy0 = xys[:2]
    xy1 = xys[2:]
    v0 = vs[:2]
    v1 = vs[2:]
    nv0 = v0 - np.dot(v0-v1,xy0-xy1)/np.sum((xy0-xy1)**2)*(xy0-xy1)
    nv1 = v1 - np.dot(v1-v0,xy1-xy0)/np.sum((xy1-xy0)**2)*(xy1-xy0)
    return np.concatenate((nv0, nv1))


def main(inits):
    t_rem = 0.0
    while True:
        bunch = solve_ivp(
            dsdt,
            (0.0, 1000.0),
            inits,
            first_step=0.01,
            dense_output=True,
            events=cev + bev
        )
        #await asyncio.sleep(t_rem / 1000)
        time.sleep(t_rem / 1000)
        if bunch.status == 1:
            t_rem = bunch.t[-1] % 10.0
            t_end = round(bunch.t[-1] - t_rem)
            n_steps = t_end // 10 + 1
            for i in range(15):
                if len(bunch.t_events[i]) > 0:
                    event_idx = i
                    #print('registered event:', i)
        elif bunch.status == 0:
            t_rem = 0.0
            t_end = 1000.0
            n_steps = 100
        #print('t_rem:', t_rem)
        #print('t_end:', t_end)
        #print('n_steps:', n_steps)
        if n_steps < 1:
            print(bunch.t_events)
            break
        for t in np.linspace(0.0, t_end, n_steps):
            yt = bunch.sol(t)
            _=canvas.coords(oval0, *ovalcoords(yt[:2]))
            _=canvas.coords(oval1, *ovalcoords(yt[2:4]))
            _=canvas.coords(oval2, *ovalcoords(yt[4:6]))
            canvas.update()
            # await asyncio.sleep(0.01)
            time.sleep(0.01)
        inits = bunch.sol(bunch.t[-1])
        if bunch.status == 1:
            #print(bunch.t_events[event_idx])
            #print(event_idx)
            try:
                inits = bunch.y_events[event_idx][0]
                if event_idx == 0:
                    vs = new_vs(inits[:4], inits[6:10])
                    inits[6:10] = vs
                elif event_idx == 1:
                    #print(np.concatenate((inits[:2], inits[4:6])),
                          #np.concatenate((inits[6:8], inits[10:12])))
                    vs = new_vs(np.concatenate((inits[:2], inits[4:6])),
                            np.concatenate((inits[6:8], inits[10:12])))
                    #print(vs)
                    inits[6:8] = vs[:2]
                    inits[10:12] = vs[2:]
                elif event_idx == 2:
                    vs = new_vs(inits[2:6], inits[8:12])
                    inits[8:12] = vs
                else:
                    #print('inits before boundary event: ', inits)
                    bev_idx = event_idx - 3
                    x_or_y = bev_idx % 2
                    bev_oval = bev_idx // 4
                    coord_idx = 2*bev_oval + x_or_y
                    v_idx = 6 + coord_idx
                    inits[v_idx] *= -1.0
                    #inits[coord_idx] += 1e-3 * inits[v_idx]
                    #print('inits after boundary event: ', inits)
            finally:
                pass
                #print(inits)


xy0 = np.array([380.0, 80.0])
xy1 = rotpts2d(xy0, 120, np.array([380.0, 380.0]))
xy2 = rotpts2d(xy1, 120, np.array([380.0, 380.0]))

inits = np.concatenate((xy0, xy1, xy2, np.random.normal(loc=0.0, scale=0.025, size=6)))

root = tk.Tk()
canvas = tk.Canvas(root, height=760, width=760, background='#000000')
canvas.pack()
oval0 = canvas.create_oval(*ovalcoords(xy0), fill='#ff0000', outline='#ff0000')
oval1 = canvas.create_oval(*ovalcoords(xy1), fill='#00ff00', outline='#00ff00')
oval2 = canvas.create_oval(*ovalcoords(xy2), fill='#0000ff', outline='#0000ff')

cev, bev = get_collision_events(), get_boundary_events()

#asyncio.run(main(inits))
main(inits)