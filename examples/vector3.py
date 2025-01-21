from engine3d.geometry.vector import Vector3D

def main():
    v = Vector3D.one()
    print("print from main: ", v)
    v.x = 5
    v.y = 12
    v.z = 2
    print("print from main: ", str(v.__dict__))

if __name__ == "__main__":
    main()