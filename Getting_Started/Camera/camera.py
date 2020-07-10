import enum
import glm
from math import sin, cos

class Camera_Movement(enum.IntEnum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3

# Default Camera Values
YAW = -90
PITCH = 0
SPEED = 2.5
SENSITIVITY = 0.1
ZOOM = 45

class Camera:
    def __init__(self,
                 position = glm.vec3(0, 0, 0),
                 up = glm.vec3(0, 1, 0),
                 yaw = YAW,
                 pitch = PITCH,
                 zoom = ZOOM,
                 sensitivity = SENSITIVITY,
                 speed = SPEED):

        # Camera Attributes
        self.Position = position
        self.Up = None
        self.Right = None
        self.WorldUp = up
        self.Front = glm.vec3(0, 0, -1)

        # euler angles
        self.Yaw = yaw
        self.Pitch = pitch

        # Camera Options
        self.MovementSpeed = speed
        self.MouseSensitivity = sensitivity
        self.Zoom = zoom

        #to get which key is pressed
        self.bool_w = False
        self.bool_a = False
        self.bool_s = False
        self.bool_d = False

        self.UpdateCameraVectors()

    def GetViewMatrix(self):
        return glm.lookAt(self.Position, self.Position + self.Front, self.Up)

    def ProcessKeyboard(self, direction, deltaTime):
        velocity = self.MovementSpeed * deltaTime
        if direction == Camera_Movement.FORWARD:
            self.Position += self.Front * velocity
        if direction == Camera_Movement.BACKWARD:
            self.Position -= self.Front * velocity
        if direction == Camera_Movement.LEFT:
            self.Position -= self.Right * velocity
        if direction == Camera_Movement.RIGHT:
            self.Position += self.Right * velocity

    def ProcessMouseMovement(self, xoffset, yoffset, constraintPitch=True):
        xoffset *= self.MouseSensitivity
        yoffset *= self.MouseSensitivity

        self.Yaw += xoffset
        self.Pitch += yoffset

        if constraintPitch:
            if self.Pitch > 89:
                self.Pitch = 89
            if self.Pitch < -89:
                self.Pitch = -89

        self.UpdateCameraVectors()

    def ProcessMouseScroll(self, yoffset):
        self.Zoom -= yoffset
        if self.Zoom < 1:
            self.Zoom = 1
        if self.Zoom > 45:
            self.Zoom = 45

    def UpdateCameraVectors(self):
        front = glm.vec3(1)
        front.x = cos(glm.radians(self.Yaw)) * cos(glm.radians(self.Pitch))
        front.y = sin(-glm.radians(self.Pitch))
        front.z = sin(glm.radians(self.Yaw)) * cos(glm.radians(self.Pitch))
        self.Front = glm.normalize(front)
        self.Right = glm.normalize(glm.cross(self.Front, self.WorldUp))
        self.Up = glm.normalize(glm.cross(self.Right, self.Front))