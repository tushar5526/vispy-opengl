#https://learnopengl.com/Getting-started/Hello-Triangle

import numpy as np
from vispy import app,gloo


vertex = """
attribute vec2 a_position;
void main (void)
{
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

fragmentWhite = """
void main()
{
    gl_FragColor = vec4(1.0,1.0,1.0,1.0);
}
"""


fragmentOrange = """
void main()
{
    gl_FragColor = vec4(1.0, 0.5, 0.2, 1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self,
                            title='Hello OpenGL',
                            keys='interactive')

        self.program1 = gloo.Program(vertex, fragmentWhite)
        self.program2 = gloo.Program(vertex, fragmentOrange)
        self.vertices = gloo.VertexBuffer(np.array([[-0.5, -0.5],
                                          [0.0, -0.5],
                                          [-0.25,  0.5],
                                           [0.5, -0.5],
                                          [0.0, -0.5],
                                          [0.25,  0.5]]).astype(np.float32))

        self.indices1 = gloo.IndexBuffer([0, 1, 2])
        self.indices2 = gloo.IndexBuffer([3, 4, 5])

        self.program1['a_position'] = self.vertices
        self.program2['a_position'] = self.vertices

        self.show()

    def on_draw(self,event):
        gloo.clear([0,0,0,0])
        self.program1.draw('triangles',self.indices1)
        self.program2.draw('triangles',self.indices2)

    def on_resize(self,event):
        gloo.set_viewport(0,0,*event.size)


if __name__ == '__main__':
    c = Canvas()
    app.run()