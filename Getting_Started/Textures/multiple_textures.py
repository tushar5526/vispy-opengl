# https://learnopengl.com/Getting-started/Textures

import numpy as np
from vispy import app, gloo, io
from vispy.gloo import Texture2D

vertex = """

attribute vec2 a_position;
attribute vec2 aTexCoord;

varying vec2 TexCoord;
varying vec3 ourColor;
void main (void)
{
    gl_Position = vec4(a_position, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment = """

varying vec2 TexCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;

void main()
{
    gl_FragColor = mix(texture2D(texture1, TexCoord), texture2D(texture2, TexCoord), 0.2);
}
"""


class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self,
                            title='Hello OpenGL',
                            keys='interactive')

        self.program = gloo.Program(vertex, fragment)

        self.vertices = gloo.VertexBuffer(np.array([
            [.5, .5],
            [.5, -.5],
            [-.5, -.5],
            [-.5, .5]]).astype(np.float32))

        self.indices = gloo.IndexBuffer([0, 1, 3, 1, 2, 3])
        self.texCoord = [(1, 1), (1, 0), (0, 0), (0, 1)]

        #flip vertically is not there in vispy yet, we can use np.flip
        self.texture1 = Texture2D(data=np.flip(io.imread('../Assets/container.jpg'), 0))

        self.texture2 = Texture2D(data=np.flip(io.imread('../Assets/smiley.jpg'), 0))

        self.program['a_position'] = self.vertices
        self.program['aTexCoord'] = self.texCoord
        self.program['texture1'] = self.texture1
        self.program['texture2'] = self.texture2

        self.timer = app.Timer('auto', self.on_timer, start=True)
        self.show()

    def on_draw(self, event):
        gloo.clear([0, 0, 0, 0])
        self.program.draw('triangles', self.indices)

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_timer(self, event):
        pass


if __name__ == '__main__':
    c = Canvas()
    app.run()