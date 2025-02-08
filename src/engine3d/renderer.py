import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pygame as pg


class RendererBase(ABC):
    width: int
    height: int
    title: str
    fps: int

    @abstractmethod
    def __init__(self, width: int, height: int, title: str, fps: int) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def handle_events(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def draw(self) -> None:
        pass


class Renderer(RendererBase):
    def __init__(self, width: int, height: int, title: str, fps: int) -> None:
        super().__init__(width, height, title, fps)

        self.width = width
        self.height = height
        self.title = title
        self.fps = fps

        pg.init()
        self.screen = pg.display.set_mode((self.width, self.height))
        self.clock = pg.time.Clock()
        self.running = True

    def run(self) -> None:
        while self.running:
            self.clock.tick(self.fps)
            self.handle_events()
            if not self.running: break
            self.draw()
            self.update()

    def handle_events(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
                pg.quit()
                sys.exit()

    def draw(self) -> None:
        self.screen.fill(pg.Color("darkslategray"))
        
    def update(self) -> None:
        pg.display.flip()

    @property
    def half_width(self) -> int:
        return self.width // 2
    
    @property
    def half_height(self) -> int:
        return self.height // 2
    
    @property
    def center(self) -> pg.math.Vector2:
        return pg.math.Vector2(self.half_width, self.half_height)