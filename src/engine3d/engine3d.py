from engine3d.renderer import Renderer


def main():
    engine3d = Renderer(width=640, height=480, title='Engine3D', fps=60)
    engine3d.run()

if __name__ == '__main__':
    main()
