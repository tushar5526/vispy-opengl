# https://learnopengl.com/Getting-started/Textures

import builtins
import numpy as np
from vispy import app, gloo, io
from vispy.gloo import Texture2D
from time import time

# python wrapper of glm
# https://pypi.org/project/PyGLM/
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

        builtins.width, builtins.height = size

        self.startTime = time()
        self.cubePositions = [glm.vec3(0.0, 0.0, 0.0),
                              glm.vec3(2.0, 5.0, -15.0),
                              glm.vec3(-1.5, -2.2, -2.5),
                              glm.vec3(-3.8, -2.0, -12.3),
                              glm.vec3(2.4, -0.4, -3.5),
                              glm.vec3(-1.7, 3.0, -7.5),
                              glm.vec3(1.3, -2.0, -2.5),
                              glm.vec3(1.5, 2.0, -2.5),
                              glm.vec3(1.5, 0.2, -1.5),
                              glm.vec3(-1.3, 1.0, -1.5)]

        self.program = gloo.Program(vertex, fragment)

        self.vertices = np.array([[-0.5, -0.5, -0.5],
                                  [0.5, -0.5, -0.5],
                                  [0.5, 0.5, -0.5],
                                  [0.5, 0.5, -0.5],
                                  [-0.5, 0.5, -0.5],
                                  [-0.5, -0.5, -0.5],

                                  [-0.5, -0.5, 0.5],
                                  [0.5, -0.5, 0.5],
                                  [0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5],
                                  [-0.5, 0.5, 0.5],
                                  [-0.5, -0.5, 0.5],

                                  [-0.5, 0.5, 0.5],
                                  [-0.5, 0.5, -0.5],
                                  [-0.5, -0.5, -0.5],
                                  [-0.5, -0.5, -0.5],
                                  [-0.5, -0.5, 0.5],
                                  [-0.5, 0.5, 0.5],

                                  [0.5, 0.5, 0.5],
                                  [0.5, 0.5, -0.5],
                                  [0.5, -0.5, -0.5],
                                  [0.5, -0.5, -0.5],
                                  [0.5, -0.5, 0.5],
                                  [0.5, 0.5, 0.5],

                                  [-0.5, -0.5, -0.5],
                                  [0.5, -0.5, -0.5],
                                  [0.5, -0.5, 0.5],
                                  [0.5, -0.5, 0.5],
                                  [-0.5, -0.5, 0.5],
                                  [-0.5, -0.5, -0.5],

                                  [-0.5, 0.5, -0.5],
                                  [0.5, 0.5, -0.5],
                                  [0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5],
                                  [-0.5, 0.5, 0.5],
                                  [-0.5, 0.5, -0.5]
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
        self.view = None
        self.projection = None

        self.program['a_position'] = self.vertices
        self.program['aTexCoord'] = self.texCoord
        self.program['texture1'] = self.texture1
        self.program['texture2'] = self.texture2

        self.timer = app.Timer('auto', self.on_timer, start=True)

        gloo.set_state(depth_test=True)
        self.show()

    def on_draw(self, event):
        # Read about depth testing and changing stated in vispy here http://vispy.org/gloo.html?highlight=set_state
        gloo.clear(color=[0.2, 0.3, 0.3, 1.0], depth=True)

        self.view = glm.mat4(1.0)
        self.view = glm.translate(self.view, glm.vec3(0.0, 0.0, -3.0))

        # to rotate camera along 45 degree or unit vector's direction
        # self.view = glm.rotate(self.view, (time() - self.startTime)* glm.radians(50), glm.vec3(1, 1, 1))

        self.projection = glm.mat4(1.0)
        self.projection = glm.perspective(glm.radians(45.0), builtins.width / builtins.height, 0.1, 100.0)

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
            i += 1
            self.model = glm.mat4(1.0)
            self.model = glm.translate(self.model, cubePosition)
            self.model = glm.rotate(self.model, glm.radians((time() - self.startTime) * glm.radians(2000) * i / 2),
                                    glm.vec3(1.0, 0.3, 0.5))
            self.model = (np.array(self.model.to_list()).astype(np.float32))
            self.model = self.model.reshape((1, self.model.shape[0] * self.model.shape[1]))
            self.program['model'] = self.model
            self.program.draw('triangles')
        self.update()

    def on_resize(self, event):
        print(*event.size)
        gloo.set_viewport(0, 0, *event.size)

    def on_timer(self, event):
        pass


if __name__ == '__main__':
    c = Canvas((800, 600))
    app.run()