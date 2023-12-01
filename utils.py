import pygame
import numpy as np
from numpy.polynomial import Polynomial
from simulation import Trail

ZHAT = np.array([0, 0, 1])
ROT90_MATRIX = np.array([[0, -1], [1, 0]])

def draw_arrow(surface, color, point1, point2, width=3, size=6):
    point1 = point1[:2]
    point2 = point2[:2]

    yhat = (point2 - point1) / np.linalg.norm(point2 - point1)
    xhat = np.dot(ROT90_MATRIX, yhat)
    pygame.draw.polygon(surface, color, [
        tuple(point1 - width/2 * xhat - width/2 * yhat),
        tuple(point1 + width/2 * xhat - width/2 * yhat),
        tuple(point2 + width/2 * xhat - size/2 * yhat),
        tuple(point2 + size * xhat - size/2 * yhat),
        tuple(point2 + size * yhat),
        tuple(point2 - size * xhat - size/2 * yhat),
        tuple(point2 - width/2 * xhat - size/2 * yhat)
    ])

"""
Euler: v^2 = G/r * 5/4
Lagrange: v^2 = G/r * 1/sqrt(3)
"""

def load_solution(sim, name):
    
    if name == "Euler 1":
        sim.n = 3
        sim.masses = np.array([1, 1, 1])
        sim.bodies = np.array([
            [[0, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [0, np.sqrt(1.25*sim.G), 0]],
            [[-1, 0, 0], [0, -np.sqrt(1.25*sim.G), 0]]
        ])
    
    elif name == "Euler 2":
        m1, m2, m3 = sim.masses

        p = Polynomial([-m2-m3, -2*m2-3*m3, -m2-3*m3, 3*m1+m2, 3*m1+2*m2, m1+m2])
        z = sorted(p.roots(), key=lambda x: abs(x.imag))[0].real

        w = np.sqrt((sim.G*m2/z**2 + sim.G*m3/(1+z)**2)/m1)
        r1 = -(m2 + (1+z)*m3)/(m1 + m2 + m3)
        r2, r3 = r1+1, r1+1+z

        sim.n = 3
        sim.masses = np.array([m1, m2, m3])
        sim.bodies = np.array([
            [[r1, 0, 0], [0, r1*w, 0]],
            [[r2, 0, 0], [0, r2*w, 0]],
            [[r3, 0, 0], [0, r3*w, 0]]
        ])
    
    elif name == "Lagrange":
        m1, m2, m3 = sim.masses

        r_cm = np.array([(m2 + 0.5*m3)/(m1 + m2 + m3), np.sqrt(0.75)*m3/(m1 + m2 + m3), 0])
        r1 = -r_cm
        r2 = [1, 0, 0] - r_cm
        r3 = [0.5, np.sqrt(0.75), 0] - r_cm

        w = np.sqrt(sim.G * (m1 + m2 + m3))

        sim.n = 3
        sim.masses = np.array([m1, m2, m3])
        sim.bodies = np.array([
            [r1, np.cross(ZHAT, r1*w)],
            [r2, np.cross(ZHAT, r2*w)],
            [r3, np.cross(ZHAT, r3*w)]
        ])
    
    elif name == "Figure-8":
        sim.G = 1
        sim.n = 3
        sim.masses = np.array([1, 1, 1])
        sim.bodies = np.array([
            [[0.97000436, -0.24308753, 0], [0.466203685, 0.43236573, 0]],
            [[-0.97000436, 0.24308753, 0], [0.466203685, 0.43236573, 0]],
            [[0, 0, 0], [-0.93240737, -0.86473146, 0]]
        ])
    
    elif name == "Custom":
        pass

    sim.trails = [Trail(2) for _ in range(sim.n)]