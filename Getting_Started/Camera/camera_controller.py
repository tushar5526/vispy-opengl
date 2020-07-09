# https://learnopengl.com/Getting-started/Textures

import builtins
import numpy as np
from vispy import app, gloo, io, use
from vispy.gloo import Texture2D
from time import time
from math import sin, cos
from vispy.gloo import gl

#python wrapper of glm
#https://pypi.org/project/PyGLM/
import glm



vertex = """

attribute vec3 a_position;
attribute vec2 aTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

varying vec2 TexCoord;
varying vec3 ourColor;
void main (void)
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment = """

varying vec2 TexCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;

void main()
{
    gl_FragColor = mix(texture2D(texture1, TexCoord), texture2D(texture2, TexCoord), 0.2f);
}
"""


class Canvas(app.Canvas):
    def __init__(self, size):
        app.Canvas.__init__(self,
                            title='Hello OpenGL',
                            keys='interactive',
                            size=size)

        use('Glfw')
        builtins.width, builtins.height = size

        self.startTime = time()
        self.first_mouse = True
        self.cubePositions = [ glm.vec3( 0.0,  0.0,  0.0),
        glm.vec3( 2.0,  5.0, -15.0),
        glm.vec3(-1.5, -2.2, -2.5),
        glm.vec3(-3.8, -2.0, -12.3),
        glm.vec3( 2.4, -0.4, -3.5),
        glm.vec3(-1.7,  3.0, -7.5),
        glm.vec3( 1.3, -2.0, -2.5),
        glm.vec3( 1.5,  2.0, -2.5),
        glm.vec3( 1.5,  0.2, -1.5),
        glm.vec3(-1.3,  1.0, -1.5)]

        self.program = gloo.Program(vertex, fragment)

        self.vertices = np.array([[-0.5, -0.5, -0.5],
                                  [0.5, -0.5, -0.5],
                                  [0.5,  0.5, -0.5],
                                  [0.5,  0.5, -0.5],
                                  [-0.5,  0.5, -0.5],
                                  [-0.5, -0.5, -0.5],

                                  [-0.5, -0.5,  0.5],
                                  [0.5, -0.5,  0.5],
                                  [0.5,  0.5,  0.5],
                                  [0.5,  0.5,  0.5],
                                  [-0.5,  0.5,  0.5],
                                  [-0.5, -0.5,  0.5],

                                  [-0.5,  0.5,  0.5],
                                  [-0.5,  0.5, -0.5],
                                  [-0.5, -0.5, -0.5],
                                  [-0.5, -0.5, -0.5],
                                  [-0.5, -0.5,  0.5],
                                  [-0.5,  0.5,  0.5],

                                  [0.5,  0.5,  0.5],
                                  [0.5,  0.5, -0.5],
                                  [0.5, -0.5, -0.5],
                                  [0.5, -0.5, -0.5],
                                  [0.5, -0.5,  0.5],
                                  [0.5,  0.5,  0.5],

                                  [-0.5, -0.5, -0.5],
                                  [0.5, -0.5, -0.5],
                                  [0.5, -0.5,  0.5],
                                  [0.5, -0.5,  0.5],
                                  [-0.5, -0.5,  0.5],
                                  [-0.5, -0.5, -0.5],

                                  [-0.5,  0.5, -0.5],
                                  [0.5,  0.5, -0.5],
                                  [0.5,  0.5,  0.5],
                                  [0.5,  0.5,  0.5],
                                  [-0.5,  0.5,  0.5],
                                  [-0.5,  0.5, -0.5]
                                  ]).astype(np.float32)
        self.texCoord = np.array([[0.0, 0.0],
                                  [1.0, 0.0],
                                  [1.0, 1.0],
                                  [1.0, 1.0],
                                  [0.0, 1.0],
                                  [0.0, 0.0],

                                  [0.0, 0.0],
                                  [1.0, 0.0],
                                  [1.0, 1.0],
                                  [1.0, 1.0],
                                  [0.0, 1.0],
                                  [0.0, 0.0],

                                  [1.0, 0.0],
                                  [1.0, 1.0],
                                  [0.0, 1.0],
                                  [0.0, 1.0],
                                  [0.0, 0.0],
                                  [1.0, 0.0],

                                  [1.0, 0.0],
                                  [1.0, 1.0],
                                  [0.0, 1.0],
                                  [0.0, 1.0],
                                  [0.0, 0.0],
                                  [1.0, 0.0],

                                  [0.0, 1.0],
                                  [1.0, 1.0],
                                  [1.0, 0.0],
                                  [1.0, 0.0],
                                  [0.0, 0.0],
                                  [0.0, 1.0],

                                  [0.0, 1.0],
                                  [1.0, 1.0],
                                  [1.0, 0.0],
                                  [1.0, 0.0],
                                  [0.0, 0.0],
                                  [0.0, 1.0]
                                  ]).astype(np.float32)

        self.texture1 = Texture2D(data=io.imread('../Textures/container.jpg', flipVertically=True))
        self.texture2 = Texture2D(data=io.imread('../Textures/smiley.jpg', flipVertically=True))

        self.model = None
        self.projection = None
        self.view = None
        self.direction = None
        self.cameraPos = glm.vec3(0, 0, 3)
        self.cameraFront = glm.vec3(0, 0, -1)
        self.cameraUp = glm.vec3(0, 1, 0)
        self.cameraSpeed = 5

        #to get key pressed behaviour
        self.bool_w = False
        self.bool_a = False
        self.bool_s = False
        self.bool_d = False

        #delta time
        self.delta_time = 0
        self.last_frame = 0

        #mouse variables
        self.last_x = None
        self.last_y = None
        self.yaw = -90
        self.pitch = 0

        self.fov = 45

        self.program['a_position'] = self.vertices
        self.program['aTexCoord'] = self.texCoord
        self.program['texture1'] = self.texture1
        self.program['texture2'] = self.texture2

        self.timer = app.Timer('auto', self.on_timer, start=True)

        gloo.set_state(depth_test=True)

        self.show()

    def on_draw(self, event):

        #Read about depth testing and changing stated in vispy here http://vispy.org/gloo.html?highlight=set_state
        gloo.clear(color=[0.2, 0.3, 0.3, 1.0], depth=True)

        #delta_time
        self.current_frame = time()
        self.delta_time = self.current_frame - self.last_frame
        self.last_frame = self.current_frame

        if self.bool_a:
            self.cameraPos -= glm.normalize(glm.cross(self.cameraFront, self.cameraUp)) * self.cameraSpeed * self.delta_time
        if self.bool_w:
            self.cameraPos += self.cameraSpeed * self.cameraFront * self.delta_time
        if self.bool_s:
            self.cameraPos -= self.cameraSpeed * self.cameraFront * self.delta_time
        if self.bool_d:
            self.cameraPos += glm.normalize(glm.cross(self.cameraFront, self.cameraUp)) * self.cameraSpeed * self.delta_time


        self.view = glm.lookAt(self.cameraPos, self.cameraPos + self.cameraFront, self.cameraUp)

        self.projection = glm.mat4(1.0)
        self.projection = glm.perspective(glm.radians(self.fov), builtins.width/builtins.height, 0.1, 100.0)

        # vispy takes numpy array in m * n matrix form
        self.view = (np.array(self.view.to_list()).astype(np.float32))
        self.projection = (np.array(self.projection.to_list()).astype(np.float32))

        # reshaping to (m, n) to (1, m*n) to support data input in vispy
        self.view = self.view.reshape((1, self.view.shape[0] * self.view.shape[1]))
        self.projection = self.projection.reshape((1, self.projection.shape[0] * self.projection.shape[1]))

        self.program['view'] = self.view
        self.program['projection'] = self.projection

        i = 0
        for cubePosition in self.cubePositions:
            self.model = glm.mat4(1.0)
            self.model = glm.translate(self.model, cubePosition)

            if i % 3 == 0:
                self.model = glm.rotate(self.model, glm.radians((time() - self.startTime) * glm.radians(2000) * i/2), glm.vec3(1.0, 0.3, 0.5))

            self.model = (np.array(self.model.to_list()).astype(np.float32))
            self.model = self.model.reshape((1, self.model.shape[0] * self.model.shape[1]))
            self.program['model'] = self.model
            self.program.draw('triangles')
            i += 1

        self.update()

    def on_key_press(self,event):
        if event.key == 'W':
            self.bool_w = True
        if event.key == 'S':
            self.bool_s = True
        if event.key == 'A':
            self.bool_a = True
        if event.key == 'D':
            self.bool_d = True

    def on_key_release(self, event):
        if event.key == 'W':
            self.bool_w = False
        if event.key == 'S':
            self.bool_s = False
        if event.key == 'A':
            self.bool_a = False
        if event.key == 'D':
            self.bool_d = False

    def on_mouse_move(self, event):
        x_pos = event.pos[0]
        y_pos = event.pos[1]

        if self.first_mouse:
            self.first_mouse = False
            self.last_x = x_pos
            self.last_y = y_pos

        x_offset = x_pos - self.last_x
        y_offset = y_pos - self.last_y
        self.last_x = x_pos
        self.last_y = y_pos

        sensitivity = 0.05
        x_offset *= sensitivity
        y_offset *= sensitivity

        self.yaw += x_offset
        self.pitch += y_offset

        if self.pitch > 89:
            self.pitch = 89
        if self.pitch < -89:
            self.pitch = -89

        direction = glm.vec3(1)
        direction.x = cos(glm.radians(self.yaw) * cos(glm.radians(self.pitch)))
        direction.y = sin(-glm.radians(self.pitch))
        direction.z = sin(glm.radians(self.yaw)) * cos(glm.radians(self.pitch))
        self.cameraFront = glm.normalize(direction)

        self.update()

    def on_mouse_wheel(self, event):
        self.fov -= event.delta[1]
        if self.fov < 1:
            self.fov = 1
        if self.fov > 45:
            self.fov = 45

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_timer(self, event):
        pass


if __name__ == '__main__':
    c = Canvas((800,600))
    app.run()