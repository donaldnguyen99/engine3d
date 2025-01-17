import numpy as np

from engine3d.object.objects import Object3D


class World:

    objects: list[Object3D]

    def __init__(self) -> None:
        self.objects = []

    def add_object(self, obj: Object3D) -> None:
        self.objects.append(obj)

    def remove_object(self, obj: Object3D) -> None:
        self.objects.remove(obj)

    def render(self) -> None:
        for obj in self.objects:
            obj.render()