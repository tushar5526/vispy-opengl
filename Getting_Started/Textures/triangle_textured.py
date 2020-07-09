#https://learnopengl.com/Getting-started/Textures

# https://learnopengl.com/Getting-started/Shaders

import numpy as np
from vispy import app, gloo, io

vertex = """

attribute vec2 a_position;
attribute vec2 aTexCoord;
attribute vec3 aColor;

varying vec2 TexCoord;

void main (void)
{
    gl_Position = vec4(a_position, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment = """

varying vec2 TexCoord;

uniform sampler2D ourTexture;

void main()
{
    gl_FragColor = texture2D(ourTexture, TexCoord);
}
"""


class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self,
                            title='Hello OpenGL',
                            keys='interactive')

        # geometry shaders require full OpenGL namespace provided by PyOpenGL

        self.program = gloo.Program(vertex, fragment)

        self.vertices = gloo.VertexBuffer(np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.0, 0.5]]).astype(np.float32))

        self.indices = gloo.IndexBuffer([0, 1, 2])

        self.texCoord = [(.0, .0), (1.0, 0.0), (0.5, 1.0)]

        self.program['a_position'] = self.vertices
        self.program['aTexCoord'] = self.texCoord
        self.program['ourTexture'] = io.imread('wall.jpg')
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