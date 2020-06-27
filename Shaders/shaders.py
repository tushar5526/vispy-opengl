from vispy import gloo
from vispy.gloo import Program

class Shader:
    def __init__(self,vertex,shader):

        self.shaderCode = None
        self.vertexCode = None

        #read the vetex shader
        with open(vertex,"r") as file:
            self.vertexCode = file.read()

        #read the fragment shader
        with open(shader,"r") as file:
            self.shaderCode = file.read()

        program = Program(vertex,shader)

    def use


