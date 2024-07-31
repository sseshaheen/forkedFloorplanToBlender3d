import bpy

def main():
    # Ensure the scene is clean before adding new objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Create a simple cube
    bpy.ops.mesh.primitive_cube_add(size=2)
    cube = bpy.context.object

    # Save the Blender file
    bpy.ops.wm.save_as_mainfile(filepath="/tmp/simple_test.blend")

if __name__ == "__main__":
    main()