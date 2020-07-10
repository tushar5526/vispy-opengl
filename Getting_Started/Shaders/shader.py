#This class holds all the vertex and frag shaders

class shader:
    def __init__(self):
        self.vertex = """

        attribute vec3 a_position;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main (void)
        {
            gl_Position = projection * view * model * vec4(a_position, 1.0);
        }
        """
        self.fragment = """

        uniform vec3 objectColor;
        uniform vec3 lightColor;

        void main()
        {
            gl_FragColor = vec4(lightColor * objectColor, 1.0);
        }
        """
        self.lightSourceVertex = """

        attribute vec3 a_position;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main (void)
        {
            gl_Position = projection * view * model * vec4(a_position, 1.0);
        }
        """
        self.lightSourceFragment = """
        void main()
        {
            gl_FragColor = vec4(1.0);
        }
        """