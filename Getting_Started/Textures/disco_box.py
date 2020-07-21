# https://learnopengl.com/Getting-started/Textures

import numpy as np
from vispy import app, gloo, io
from vispy.gloo import Texture2D

vertex = """

attribute vec2 a_position;
attribute vec2 aTexCoord;
attribute vec3 aColor;

varying vec2 TexCoord;
varying vec3 ourColor;
void main (void)
{
    gl_Position = vec4(a_position, 0.0, 1.0);
    TexCoord = aTexCoord;
    ourColor = aColor;
}
"""

fragment = """

varying vec2 TexCoord;
varying vec3 ourColor;

uniform sampler2D ourTexture;

void main()
{
    gl_FragColor = texture2D(ourTexture, TexCoord) * vec4(ourColor, 1.0);
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
        self.texture = Texture2D(data=io.imread('../Assets/container.jpg'),
                                 interpolation='nearest',
                                 wrapping='repeat')

        self.color = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [1.0, 1.0, 0.0]]).astype(np.float32)

        self.program['a_position'] = self.vertices
        self.program['aTexCoord'] = self.texCoord
        self.program['ourTexture'] = self.texture
        self.program['aColor'] = self.color

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