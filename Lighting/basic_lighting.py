# https://learnopengl.com/Lighting/Basic-Lighting

import builtins
import numpy as np
from vispy import app, gloo
from time import time
from Getting_Started.Camera.camera import Camera, Camera_Movement
from math import sin

#python wrapper of glm
#https://pypi.org/project/PyGLM/
import glm

vertex = """

attribute vec3 a_position;
attribute vec3 aNormal;

varying vec3 Normal;
varying vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main (void)
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    Normal = aNormal;
    FragPos = vec3(model * vec4(a_position, 1.0));
}
"""
fragment = """

uniform vec3 lightPos;
uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 viewPos;

varying vec3 FragPos;
varying vec3 Normal;

void main()
{
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 256);
    vec3 specular = specularStrength * spec * lightColor;
    
    vec3 result = (ambient + diffuse + specular) * objectColor;
    gl_FragColor = vec4(result, 1.0);
}
"""

lightSourceVertex =  """

attribute vec3 a_position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main (void)
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
}
"""
lightSourceFragment = """
void main()
{
    gl_FragColor = vec4(1.0);
}
"""


class Canvas(app.Canvas):
    def __init__(self, size):
        app.Canvas.__init__(self,
                            title='Hello OpenGL',
                            keys='interactive',
                            size=size)

        # vispy wrapper of glfw dont have the wrapper of this function yet, I am opening a PR for this
        # by the time we can use this
        self._app.native.glfwSetInputMode(self.native._id, self._app.native.GLFW_CURSOR, self._app.native.GLFW_CURSOR_DISABLED)

        builtins.width, builtins.height = size

        #camera instance
        self.camera = Camera(position=glm.vec3(0, 0, 3), sensitivity=0.2)

        self.startTime = time()
        self.first_mouse = True
        self.lightPos = [0, 0, 0]

        self.program = gloo.Program(vertex, fragment)
        self.programLightSource = gloo.Program(lightSourceVertex, lightSourceFragment)

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
        self.aNormal = np.array([
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]).astype(np.float32)

        """self.texCoord = np.array([[0.0, 0.0],
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
                                  ]).astype(np.float32)"""

        self.model = None
        self.projection = None
        self.view = None

        #delta time
        self.delta_time = 0
        self.last_frame = 0

        #mouse variables
        self.last_x = None
        self.last_y = None

        self.timer = app.Timer('auto', self.on_timer, start=True)
        gloo.set_state(depth_test=True)

        self.show()

    def on_draw(self, event):

        #Read about depth testing and changing stated in vispy here http://vispy.org/gloo.html?highlight=set_state
        gloo.clear(color=[0, 0, 0, 1.0], depth=True)

        #delta_time
        self.current_frame = time()
        self.delta_time = self.current_frame - self.last_frame
        self.last_frame = self.current_frame

        if self.camera.bool_a:
            self.camera.ProcessKeyboard(Camera_Movement.LEFT, self.delta_time)
        if self.camera.bool_w:
            self.camera.ProcessKeyboard(Camera_Movement.FORWARD, self.delta_time)
        if self.camera.bool_s:
            self.camera.ProcessKeyboard(Camera_Movement.BACKWARD, self.delta_time)
        if self.camera.bool_d:
            self.camera.ProcessKeyboard(Camera_Movement.RIGHT, self.delta_time)

        self.view = self.camera.GetViewMatrix()
        self.projection = glm.perspective(glm.radians(self.camera.Zoom), builtins.width/builtins.height, 0.1, 100.0)

        # vispy takes numpy array in m * n matrix form
        self.view = (np.array(self.view.to_list()).astype(np.float32))
        self.projection = (np.array(self.projection.to_list()).astype(np.float32))

        # reshaping to (m, n) to (1, m*n) to support data input in vispy
        self.view = self.view.reshape((1, self.view.shape[0] * self.view.shape[1]))
        self.projection = self.projection.reshape((1, self.projection.shape[0] * self.projection.shape[1]))

        self.model = glm.mat4(1.0)
        self.model = glm.translate(self.model, self.lightPos)
        self.model = (np.array(self.model.to_list()).astype(np.float32))
        self.model = self.model.reshape((1, self.model.shape[0] * self.model.shape[1]))

        #drawing light source
        self.programLightSource['model'] = self.model
        self.programLightSource['view'] = self.view
        self.programLightSource['projection'] = self.projection
        self.programLightSource['a_position'] = self.vertices

        self.programLightSource.draw('triangles')

        # drawing normal cube
        self.program['view'] = self.view
        self.program['projection'] = self.projection
        self.program['a_position'] = self.vertices * 5
        self.program['aNormal'] = self.aNormal
        self.program['viewPos'] = self.camera.Position
        self.program['lightPos'] = self.lightPos
        self.program['objectColor'] = [1, 0.5, 0.31]
        self.program['lightColor'] = [1, 1, 1]

        self.model = glm.mat4(1.0)
        # rotate the cube if you want
        #self.model = glm.rotate(self.model, glm.radians((time() - self.startTime) * 10), glm.vec3(0,1.5,1))
        self.model = glm.translate(self.model, glm.vec3(0,-5,0))
        self.model = (np.array(self.model.to_list()).astype(np.float32))
        self.model = self.model.reshape((1, self.model.shape[0] * self.model.shape[1]))

        self.program['model'] = self.model
        self.program.draw('triangles')


        self.update()

    def on_key_press(self,event):
        if event.key == 'W':
            self.camera.bool_w = True
        if event.key == 'S':
            self.camera.bool_s = True
        if event.key == 'A':
            self.camera.bool_a = True
        if event.key == 'D':
            self.camera.bool_d = True

    def on_key_release(self, event):
        if event.key == 'W':
            self.camera.bool_w = False
        if event.key == 'S':
            self.camera.bool_s = False
        if event.key == 'A':
            self.camera.bool_a = False
        if event.key == 'D':
            self.camera.bool_d = False

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
        self.camera.ProcessMouseMovement(x_offset, y_offset)

        self.update()

    def on_mouse_wheel(self, event):
        self.camera.ProcessMouseScroll(-event.delta[1])

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_timer(self, event):
        pass


if __name__ == '__main__':
    c = Canvas((800,600))
    app.run()