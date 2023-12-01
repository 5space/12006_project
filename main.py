import numpy as np
import time

import pygame
import pygame_gui

# GUI stuff
COLORS = [(255, 0, 0),
          (0, 255, 0),
          (0, 0, 255)]

TOPRIGHT_ANCHORS = {"left": "right",
                    "right": "right",
                    "top": "top",
                    "bottom": "top"}

BOTTOMRIGHT_ANCHORS = {"left": "right",
                       "right": "right",
                       "top": "bottom",
                       "bottom": "bottom"}

BOTTOMLEFT_ANCHORS = {"left": "left",
                      "right": "left",
                      "top": "bottom",
                      "bottom": "bottom"}

# for debugging purposes, makes numpy actually show you the whole number
np.set_printoptions(precision=20)

from simulation import Simulation
from utils import draw_arrow, load_solution

sim = Simulation(G=1)

sim.add_body(1, [0, 0, 0], [0, 0, 0])
sim.add_body(1, [0, 0, 0], [0, 0, 0])
sim.add_body(1, [0, 0, 0], [0, 0, 0])
load_solution(sim, "Euler 1")

pygame.init()

WIDTH = 800
HEIGHT = 640
ZOOM = 180

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True

ui_manager = pygame_gui.UIManager((WIDTH, HEIGHT), "theme.json")

algorithm_label = pygame_gui.elements.UILabel(pygame.Rect(5, 5, 80, 25), "Algorithm:", ui_manager)
algorithm_ddl = pygame_gui.elements.UIDropDownMenu(["Euler", "Mod. Euler", "SI Euler", "Runge-Kutta"], "SI Euler", pygame.Rect(90, 5, 120, 25), ui_manager)

solution_label = pygame_gui.elements.UILabel(pygame.Rect(5, 35, 80, 25), "Solution:", ui_manager)
solution_ddl = pygame_gui.elements.UIDropDownMenu(["[Custom]", "Euler 1", "Euler 2", "Lagrange", "Figure-8"], "Euler 1", pygame.Rect(90, 35, 120, 25), ui_manager)

bump_button = pygame_gui.elements.UIButton(pygame.Rect(5, 65, 80, 25), "Bump", ui_manager)

grav_label = pygame_gui.elements.UILabel(pygame.Rect(-280, 5, 72, 25), " G = {0:.2f}".format(sim.G), ui_manager, anchors=TOPRIGHT_ANCHORS)
grav_slider = pygame_gui.elements.UIHorizontalSlider(pygame.Rect(-205, 5, 200, 25), 100*sim.G, (0, 200), ui_manager, anchors=TOPRIGHT_ANCHORS)

angmom_label = pygame_gui.elements.UILabel(pygame.Rect(-280, 35, 200, 25), "ΣL = 0.00", ui_manager, anchors=TOPRIGHT_ANCHORS)
linmom_label = pygame_gui.elements.UILabel(pygame.Rect(-280, 65, 200, 25), "Σp = 0.00", ui_manager, anchors=TOPRIGHT_ANCHORS)
energy_label = pygame_gui.elements.UILabel(pygame.Rect(-280, 95, 200, 25), "ΣE = 0.00", ui_manager, anchors=TOPRIGHT_ANCHORS)

playpause_button = pygame_gui.elements.UIButton(pygame.Rect(-65, -30, 60, 25), "Pause", ui_manager, anchors=BOTTOMRIGHT_ANCHORS)

mass_label = pygame_gui.elements.UILabel(pygame.Rect(5, -30, 40, 25), "Mass:", ui_manager, anchors=BOTTOMLEFT_ANCHORS)
mass_box = pygame_gui.elements.UITextEntryLine(pygame.Rect(50, -30, 50, 25), ui_manager, anchors=BOTTOMLEFT_ANCHORS)

pos_label = pygame_gui.elements.UILabel(pygame.Rect(105, -30, 75, 25), "Position:", ui_manager, anchors=BOTTOMLEFT_ANCHORS)
posx_box = pygame_gui.elements.UITextEntryLine(pygame.Rect(185, -30, 55, 25), ui_manager, anchors=BOTTOMLEFT_ANCHORS)
posy_box = pygame_gui.elements.UITextEntryLine(pygame.Rect(240, -30, 55, 25), ui_manager, anchors=BOTTOMLEFT_ANCHORS)

vel_label = pygame_gui.elements.UILabel(pygame.Rect(300, -30, 75, 25), "Velocity:", ui_manager, anchors=BOTTOMLEFT_ANCHORS)
velx_box = pygame_gui.elements.UITextEntryLine(pygame.Rect(380, -30, 55, 25), ui_manager, anchors=BOTTOMLEFT_ANCHORS)
vely_box = pygame_gui.elements.UITextEntryLine(pygame.Rect(435, -30, 55, 25), ui_manager, anchors=BOTTOMLEFT_ANCHORS)

update_button = pygame_gui.elements.UIButton(pygame.Rect(495, -30, 60, 25), "Update", ui_manager, anchors=BOTTOMLEFT_ANCHORS)

BODY_SPECIFIC_ELEMENTS = [mass_label, mass_box, pos_label, posx_box, posy_box, vel_label, velx_box, vely_box, update_button]
for e in BODY_SPECIFIC_ELEMENTS:
    e.hide()

ui_visible = True

def toggle_running():
    sim.running = not sim.running
    if sim.running:
        playpause_button.set_text("Pause")
    else:
        playpause_button.set_text("Play")

def radius_of_body(mass):
    return (24*mass + 120)/(mass + 20)

selected_body = -1
hovered_body = -1
current_algorithm = sim.step_sieuler
current_solution = ""

algorithms = {
    "Euler": sim.step_euler,
    "Mod. Euler": sim.step_modifiedeuler,
    "SI Euler": sim.step_sieuler,
    "Runge-Kutta": sim.step_rungekutta
}

def screenshot_screen():
    filename = "screenshot" + time.strftime("%Y%m%d%H%M%S") + ".png"
    pygame.image.save(screen, filename)

while running:
    ui_dt = clock.tick(60)/1000.0
    phys_dt = 0.016

    hovered_body = -1
    mouse_pos = np.array([*pygame.mouse.get_pos(), 0]) - [WIDTH/2, HEIGHT/2, 0]
    mouse_pos /= ZOOM
    for i in range(sim.n):
        if ZOOM * np.linalg.norm(mouse_pos - sim.bodies[i][0]) <= 3 * radius_of_body(sim.masses[i]):
            hovered_body = i
            break
    
    if sim.running:
        current_algorithm(phys_dt)
    
    for event in pygame.event.get():
        ui_manager.process_events(event)
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.size
            ui_manager.set_window_resolution((WIDTH, HEIGHT))
            ui_manager.root_container.set_dimensions((WIDTH, HEIGHT))
        
        elif event.type == pygame.USEREVENT:
            if event.ui_element == grav_slider:
                sim.G = event.value / 100
                grav_label.set_text(" G = {0:.2f}".format(sim.G))
            elif event.ui_element == algorithm_ddl and event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                current_algorithm = algorithms[event.text]
            elif event.ui_element == solution_ddl and event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                load_solution(sim, event.text)
            elif event.ui_element == playpause_button and event.user_type == pygame_gui.UI_BUTTON_START_PRESS:
                toggle_running()
            elif event.ui_element == update_button and event.user_type == pygame_gui.UI_BUTTON_START_PRESS:
                if selected_body >= 0:
                    sim.bodies[selected_body][0][0] = float(posx_box.get_text())
                    sim.bodies[selected_body][0][1] = float(posy_box.get_text())
                    sim.bodies[selected_body][1][0] = float(velx_box.get_text())
                    sim.bodies[selected_body][1][1] = float(vely_box.get_text())
                    sim.masses[selected_body] = float(mass_box.get_text())
            elif event.ui_element == bump_button and event.user_type == pygame_gui.UI_BUTTON_START_PRESS:
                sim.bump()
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                ui_visible = not ui_visible
            elif event.key == pygame.K_SPACE:
                toggle_running()
            elif event.key == pygame.K_s:
                screenshot_screen()
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if (hovered_body >= 0 and event.button == pygame.BUTTON_LEFT) or (hovered_body < 0 and event.button == pygame.BUTTON_RIGHT):
                selected_body = hovered_body
                if selected_body >= 0:
                    for e in BODY_SPECIFIC_ELEMENTS:
                        e.show()
                    pos, vel = sim.bodies[selected_body]
                    posx_box.set_text(str(pos[0]))
                    posy_box.set_text(str(pos[1]))
                    velx_box.set_text(str(vel[0]))
                    vely_box.set_text(str(vel[1]))
                    mass_box.set_text(str(sim.masses[selected_body]))
                else:
                    for e in BODY_SPECIFIC_ELEMENTS:
                        e.hide()
        
        elif event.type == pygame.MOUSEWHEEL:
            ZOOM *= 1.05 ** event.y
    
    if ui_visible:
        ui_manager.update(ui_dt)
        angmom_label.set_text("ΣL = {0:.3f}".format(sim.angular_momentum()).ljust(25))
        x, y, _ = sim.linear_momentum()
        linmom_label.set_text("Σp = ({0:.3f}, {1:.3f})".format(x, y).ljust(25))
        energy_label.set_text("ΣE = {0:.3f}".format(sim.energy()).ljust(25))

    # screen.fill((0, 0, 0))
    screen.fill((255, 255, 255))
    canvas = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    canvas.fill((0, 0, 0, 0))

    for i in range(sim.n):
        trail = sim.trails[i]
        points = trail.points
        if len(points) == 0: continue
        p2 = points[0][1] * ZOOM + np.array([WIDTH/2, HEIGHT/2, 0])
        for j in range(1, len(points)):
            p1, p2 = p2, points[j][1] * ZOOM + np.array([WIDTH/2, HEIGHT/2, 0])
            pygame.draw.line(canvas, (*COLORS[i], 255 + int(255/trail.time * points[j][0])), tuple(p1)[:2], tuple(p2)[:2], 2)

    for i, (pos, vel) in enumerate(sim.bodies):
        screen_pos = pos * ZOOM + np.array([WIDTH/2, HEIGHT/2, 0])
        x, y, z = screen_pos
        radius = radius_of_body(sim.masses[i])

        draw_arrow(canvas, (*COLORS[i], 128), screen_pos, screen_pos + vel * ZOOM / 4)
        pygame.draw.circle(canvas, COLORS[i], (x, y), radius)
    
    screen.blit(canvas, (0, 0))
    
    if ui_visible:
        ui_manager.draw_ui(screen)

    pygame.display.flip()

pygame.quit()