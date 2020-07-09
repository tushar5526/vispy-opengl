#Next step is to make a shader class, but vispy already provides wrapper for OpenGL functions
#so there is no need for it.

#https://learnopengl.com/Getting-started/Shaders

import numpy as np
from vispy import app,gloo
from time import time
from math import sin,cos

vertex = """

attribute vec2 a_position;
attribute vec3 a_color;

varying vec3 color;

void main (void)
{
    gl_Position = vec4(a_position, 0.0, 1.0);
    color = a_color;
}
"""

fragment = """

varying vec3 color;

void main()
{
    gl_FragColor = vec4(color, 1.0);
}
"""


class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self,
                            title='Hello OpenGL',
                            keys='interactive')

        # geometry shaders require full OpenGL namespace provided by PyOpenGL

        self.program = gloo.Program(vertex, fragment)

        self.vertices = gloo.VertexBuffer(np.array([[-0.5, -0.5],
                                          [0.5, -0.5],
                                          [0.0,  0.5]]).astype(np.float32))

        self.indices = gloo.IndexBuffer([0, 1, 2, 3, 4, 5])

        self.program['a_position'] = self.vertices

        self.timer = app.Timer('auto',self.on_timer,start=True)
        self.show()

    def on_draw(self,event):
        gloo.clear([0,0,0,0])
        self.program.draw('triangles',self.indices)

    def on_resize(self,event):
        gloo.set_viewport(0,0,*event.size)

    def on_timer(self,event):
        #print((sin(time()) / 2.0) + 0.5)
        self.program['a_color'] = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]).astype(np.float32)

        self.update()

if __name__ == '__main__':
    c = Canvas()
    app.run()