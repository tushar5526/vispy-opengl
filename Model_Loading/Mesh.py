import glm
from collections import  namedtuple

Vertex = namedtuple('Vertex','Position Normal TexCoords')
Texture = namedtuple(('Texture', 'id type'))

class Mesh:
    def __init__(self, vertices, indices, textures):
        self.vertices = vertices
        self.indices = indices
        self.textures = textures

        self.setupMesh()

    def setupMesh(self):
        pass

    def Draw(self, program):
        diffuseNr = 1
        specularNr = 1

        for i in range(0, len(self.textures)):
            name = self.textures[i].type
            number = None
            if name == 'texture_diffuse':
                number = str(diffuseNr)
                diffuseNr += 1
            elif name == 'texture_specular':
                number = str(specularNr)
                specularNr += 1

            program['material.' + str(name) + str(number)] = i

        #draw mesh





