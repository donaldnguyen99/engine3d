import numpy as np
import pygame as pg

from engine3d.renderer import Renderer
from engine3d.math.vector import Vector3D

class Triangle:
    def __init__(self, vertices: np.ndarray) -> None:
        self.vertices = [Vector3D(*vertex) for vertex in vertices]  # Convert each vertex to Vector4D([*
        self.color = (255, 255, 255)
        self.edges = np.array([[0, 1], [1, 2], [2, 0]])
        self.angle = 0

    def draw(self, renderer: Renderer) -> None:
        for edge in self.edges:
            v0 = self.vertices[edge[0]] - Vector3D(0.5, 0.5, 0.5) # Center the triangle
            v1 = self.vertices[edge[1]] - Vector3D(0.5, 0.5, 0.5) # Center the triangle
            # Rotate the vertices around the y-axis
            v0 = v0.rotate(self.angle, Vector3D(0, 1, 1))
            v1 = v1.rotate(self.angle, Vector3D(0, 1, 1))
            # Currently using Orthographic mapping: just use the x and y coordinates
            vertex0 = pg.math.Vector2(*v0.xy.to_array())   # Convert to Vector2D and add center
            vertex1 = pg.math.Vector2(*v1.xy.to_array())   # Convert to Vector2D and add center
            pg.draw.line(
                renderer.screen, 
                (255, 0, 0), 
                (vertex0) * 100 + renderer.center, 
                vertex1 * 100 + renderer.center)
    
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