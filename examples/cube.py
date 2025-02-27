import numpy as np
import pygame as pg

from engine3d.renderer import Renderer
from engine3d.math.vector import Vector3D, Vector4D
from engine3d.math.matrix import TranslationMatrix4x4Homogeneous, ScaleMatrix4x4Homogeneous, RotationMatrix4x4Homogeneous, PerspectiveProjectionMatrix4x4Homogeneous

class Triangle:
    def __init__(self, vertices: np.ndarray) -> None:
        self.vertices = [Vector3D(*vertex) for vertex in vertices]  # Convert each vertex to Vector4D([*
        self.color = (255, 255, 255)
        self.edges = np.array([[0, 1], [1, 2], [2, 0]])
        self.angle: float = 0

    def draw(self, renderer: Renderer) -> None:
        for edge in self.edges:
            v0 = Vector4D(*self.vertices[edge[0]], 1)
            v1 = Vector4D(*self.vertices[edge[1]], 1)
            # Rotate the vertices around the y-axis
            model_matrix = RotationMatrix4x4Homogeneous(self.angle, Vector3D(1, 0.1, 0)) @\
                ScaleMatrix4x4Homogeneous(Vector3D(10, 10, 10)) @\
                TranslationMatrix4x4Homogeneous(Vector3D(-0.5, -0.5, -0.5))
            view_matrix = RotationMatrix4x4Homogeneous(0.02 * np.pi, Vector3D(0, 1, 0)) @\
                TranslationMatrix4x4Homogeneous(Vector3D(0, 0, -15))
            modelview_matrix = view_matrix @ model_matrix
            projection_matrix = PerspectiveProjectionMatrix4x4Homogeneous(70 * np.pi/180, renderer.width/renderer.height, 0.1, 1000)
            v0 = projection_matrix @ modelview_matrix @ v0
            v1 = projection_matrix @ modelview_matrix @ v1
            # Rotate the vertices around the x-axis
            # v0 = v0.rotate(self.angle, Vector3D(0, 1, 1))
            # v1 = v1.rotate(self.angle, Vector3D(0, 1, 1))
            # Currently using Orthographic mapping: just use the x and y coordinates
            # vertex0 = pg.math.Vector2(*v0.xy.to_array())   # Convert to Vector2D and add center
            # vertex1 = pg.math.Vector2(*v1.xy.to_array())   # Convert to Vector2D and add center

            # Perspective projection
            v0c = v0.homogeneous_to_cartesian()
            v1c = v1.homogeneous_to_cartesian()
            vertex0 = pg.math.Vector2(*(v0c.xy).to_array())   # Convert to Vector2D and add center
            vertex1 = pg.math.Vector2(*(v1c.xy).to_array())   # Convert to Vector2D and add center

            screen_pos0 = pg.math.Vector2(
                vertex0.x * renderer.half_width + renderer.center.x,
                renderer.height - (vertex0.y * renderer.half_height + renderer.center.y),
            )
            screen_pos1 = pg.math.Vector2(
                vertex1.x * renderer.half_width + renderer.center.x,
                renderer.height - (vertex1.y * renderer.half_height + renderer.center.y),
            )

            
            pg.draw.line(
                renderer.screen, 
                (int(v0c.z < 2) * 255, 0, 0), 
                screen_pos0, 
                screen_pos1)
    
    def update(self) -> None:
        self.angle += 0.01
        self.angle %= 2 * np.pi

class MyRenderer(Renderer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.objects = [
            # Triangles of a cube
            Triangle(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])), # front 1
            Triangle(np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])), # front 2
            Triangle(np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]])), # back 1
            Triangle(np.array([[1, 0, 1], [1, 1, 1], [0, 1, 1]])), # back 2
            Triangle(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])), # bottom 1
            Triangle(np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1]])), # bottom 2
            Triangle(np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]])), # top 1
            Triangle(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])), # top 2
            Triangle(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])), # left 1
            Triangle(np.array([[0, 1, 0], [0, 1, 1], [0, 0, 1]])), # left 2
            Triangle(np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]])), # right 1
            Triangle(np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1]])), # right 2
        ]

    def draw(self) -> None:
        super().draw()
        for obj in self.objects:
            obj.draw(self)

    def update(self) -> None:
        for obj in self.objects:
            obj.update()
        super().update()

if __name__ == "__main__":
    renderer = MyRenderer(800, 600, "Cube Renderer", 60)
    renderer.run()