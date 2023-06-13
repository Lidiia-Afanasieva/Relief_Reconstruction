import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from camera import *
from ORB import *
from random import random


gen = mender_cam()
# next(gen)
points = np.array(next(gen))

estimatedPoints = np.array([(random(),random(),random()) for _ in range(70)])

def drawAxes():
    glColor((255, 255, 255))
    for v in np.array([(1,0,0), (0,1,0), (0,0,1)]):
        glColor(255*v)
        glBegin(GL_LINES)
        glVertex3fv((0,0,0)); glVertex3fv(v)
        glEnd()

def drawCubePts(points):
    glColor((255, 255, 255))
    glBegin(GL_POINTS)
    for v in points:
        # print()
        glVertex2i(int(v[0]), int(v[1]))
    glEnd()

def drawEstimatedPts():
    glColor((255, 0, 0))
    glBegin(GL_POINTS)
    print('LEN V IN DRAWESTPOINTS: ', len(estimatedPoints))
    for v in estimatedPoints:
        glVertex3fv(v)
    glEnd()



def main():
    global estimatedPoints
    pygame.init()
    display = (1280, 960) #______________________________________________
    # display = (960, 1280)
    screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    cam=Camera()

    # MODE="manual"
    MODE="opengl"

    i=0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        cam.x = 1
        cam.z = 1

        if(MODE=="opengl"):
            gluPerspective(123, (display[0] / display[1]), 0.1, 50.0) #___
            z, x, y = -cam.x, -cam.y, -cam.z
            glTranslatef(x, y, z)
            glMultMatrixf(np.transpose(cam.swapCS))
            glRotatef(cam.yaw, 0, 0, 1)    #z
            glRotatef(cam.pitch, 0, 1, 0)  #y
            glRotatef(cam.roll, 1, 0, 0)   #x
            MP = glGetFloatv(GL_PROJECTION_MATRIX)
            if i==0: print(MP)

        screenPts = next(gen)
        # print('screenPts: ', screenPts)
        # gener = _get_match_pixel('/rand_blanket.mp4')
        # # for i in range(3):
        # #     print(next(gener))
        # screenPts = next(gener)

        drawAxes()

        glPointSize(7.0)
        drawCubePts(screenPts)
        glPointSize(4.0)
        
        drawEstimatedPts()
        print("_________________")
        print(f'SCREEN POINTS: {screenPts}')
        print("_________________")
        print("_________________")
        print(f'ESTIMATED POINTS: {estimatedPoints}')
        print("_________________")

        estimatedPoints = cam.doGradDescent(screenPts, estimatedPoints)
        pygame.display.flip()
        pygame.time.wait(20)

        # cam.x += 0.1
        cam.z += 0.01

        cam.yaw += 1
        # cam.pitch+=1
        # cam.roll+=1
        i+=1

main()