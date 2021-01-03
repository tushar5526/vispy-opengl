# https://learnopengl.com/Getting-started/Textures

import numpy as np
from vispy import app, gloo, io
from vispy.gloo import Texture2D
from time import time
from math import sin
#python wrapper of glm
#https://pypi.org/project/PyGLM/
import glm



vertex = """

attribute vec2 a_position;
attribute vec2 aTexCoord;
uniform mat4 transform;

varying vec2 TexCoord;
varying vec3 ourColor;
void main (void)
{
    gl_Position = transform * vec4(a_position, 0.0, 1.0);
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
    def __init__(self):
        app.Canvas.__init__(self,
                            title='Hello OpenGL',
                            keys='interactive')

        self.startTime = time()
        self.program = gloo.Program(vertex, fragment)

        self.vertices = gloo.VertexBuffer(np.array([
            [.5, .5],
            [.5, -.5],
            [-.5, -.5],
            [-.5, .5]]).astype(np.float32))

        self.indices = gloo.IndexBuffer([0, 1, 3, 1, 2, 3])
        self.texCoord = [(1, 1), (1, 0), (0, 0), (0, 1)]
        self.texture1 = Texture2D(data=io.imread('../Assets/container.jpg', flipVertically=True))
        self.texture2 = Texture2D(data=io.imread('../Assets/smiley.jpg', flipVertically=True))

        #trans stores matrix in glm format
        #_trans stores matrix in numpy format
        self.trans = None
        self._trans = None

        self.program['a_position'] = self.vertices
        self.program['aTexCoord'] = self.texCoord
        self.program['texture1'] = self.texture1
        self.program['texture2'] = self.texture2



        self.timer = app.Timer('auto', self.on_timer, start=True)
        self.show()

    def on_draw(self, event):
        gloo.clear([0.2, 0.3, 0.3, 1.0])
        self.program.draw('triangles', self.indices)

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_timer(self, event):
        #we can do this in on_draw too, we will use it in exercise2's solution
        self.trans = glm.mat4(1.0)
        self.trans = glm.translate(self.trans, glm.vec3(0.5, -.5, .0))
        self.trans = glm.rotate(self.trans, time() - self.startTime,glm.vec3(0.0, 0.0, 1.0))

        #convering to numpy arrray to support vispy
        self._trans = (np.array(self.trans.to_list()).astype(np.float32))
        # reshaping to (m, n) to (1, m*n)
        self._trans = self._trans.reshape((1, self._trans.shape[0] * self._trans.shape[1]))

        self.program['transform'] = self._trans
        self.update()


if __name__ == '__main__':
    c = Canvas()
    app.run()