This is the python implementation of the OpenGL tutorial on www.learnopengl.com

Vispy stable release dont support #330 version of GLSL, so we are using older versions 
where `out` and `in` are replaced by `varying` keywords, and `attributes` are used instead of `layout`

You have to setup vispy for your project or use the `requirements.txt`

**Vispy do not support structs in fragment shader** which  means something like `self.program['material.ambient'] is not gonna work.

Will be continuing the tutorials on pyopengl now onwards.