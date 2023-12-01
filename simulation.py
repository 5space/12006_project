import numpy as np
from random import random

# class that runs the physics simulation
class Simulation:

    def __init__(self, masses=[], bodies=[], G=1):
        self.masses = masses
        # IMPORTANT: each element of self.bodies is a 3x2 array structured like [[x, y, z], [v_x, v_y, v_z]] 
        self.bodies = bodies
        self.G = G

        self.t = 0
        self.n = len(self.masses)
        self.running = True

        self.trails = []
    
    def add_body(self, mass, pos, vel):
        body = np.array([pos, vel])
        self.masses.append(mass)
        self.bodies.append(body)
        self.trails.append(Trail(2))
        self.n += 1
    
    def set_body(self, index, mass, pos, vel):
        body = np.array([pos, vel])
        self.masses[index] = mass
        self.bodies[index] = body
        self.trails[index].clear()
    
    def remove_body(self, index):
        self.masses.pop(index)
        self.bodies.pop(index)
        self.trails.pop(index)
        self.n -= 1
    
    def update_trails(self, dt):
        for i in range(self.n):
            self.trails[i].stack(self.bodies[i][0], dt)
    
    def bump(self, c=0.00001):
        for b in self.bodies:
            b[0] += np.random.uniform(-c, c)
    
    # acceleration vector
    def dv_dt(self, i, body=None):
        if body is None: body = self.bodies[i]

        F = np.zeros(self.n)
        for j in range(self.n):
            if i == j: continue
            offset = self.bodies[j][0] - self.bodies[i][0]
            F += self.G * self.masses[j] * offset / np.linalg.norm(offset) ** 3
        return F

    def ode(self, i, body=None):
        if body is None: body = self.bodies[i]
        return np.array([body[1], self.dv_dt(i, body)])
    
    def step_euler(self, dt):
        new = []
        for i in range(self.n):
            new.append(self.bodies[i] + self.ode(i) * dt)
        self.bodies = new
        self.t += dt
        self.update_trails(dt)
     
    def step_sieuler(self, dt):
        new = []
        for i in range(self.n):
            y0 = self.bodies[i] + self.ode(i) * dt
            r1, v1 = y0
            r1 += self.ode(i)[1] * dt * dt
            new.append(np.array([r1, v1]))
        self.bodies = new
        self.t += dt
        self.update_trails(dt)

    def step_modifiedeuler(self, dt):
        new = []
        for i in range(self.n):
            y0 = self.bodies[i]
            k1 = self.ode(i, y0)
            k2 = self.ode(i, y0 + dt*k1)
            new.append(y0 + dt/2 * (k1 + k2))
        self.bodies = new
        self.t += dt
        self.update_trails(dt)
    
    def step_rungekutta(self, dt):
        new = []
        for i in range(self.n):
            y0 = self.bodies[i]
            k1 = self.ode(i, y0)
            k2 = self.ode(i, y0 + dt*k1/2)
            k3 = self.ode(i, y0 + dt*k2/2)
            k4 = self.ode(i, y0 + dt*k3)
            new.append(y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4))
        self.bodies = new
        self.t += dt
        self.update_trails(dt)
    
    def center_of_mass(self):
        return sum(self.masses[i] * self.bodies[i][0] for i in range(self.n)) / sum(self.masses)
    
    def energy(self):
        e = 0
        amt = self.n
        for i in range(amt):
            e += self.masses[i] * np.linalg.norm(self.bodies[i][1]) ** 2 / 2
            for j in range(i+1, amt):
                e -= self.G * self.masses[i] * self.masses[j] / np.linalg.norm(self.bodies[j][0] - self.bodies[i][0])
        return e
    
    def linear_momentum(self):
        return sum(self.masses[i] * self.bodies[i][1] for i in range(self.n))
    
    def angular_momentum(self, reference_frame=np.zeros([2, 3])):
        return sum(self.masses[i] * np.cross(*(self.bodies[i] - reference_frame)) for i in range(self.n))[2]
    
    def to_xyz(self):
        return tuple([b[0][i] for b in self.bodies] for i in range(3))


class Trail:

    def __init__(self, time=1):
        self.time = time
        self.points = []
    
    def stack(self, point, dt):
        for p in self.points:
            p[0] -= dt
        self.points.append([0, point])
        while self.points[0][0] <= -self.time: self.points.pop(0)
    
    def clear(self):
        self.points = []