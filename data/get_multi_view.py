import bpy
import math
import os

# .\blender.exe -b --python "E:/Terry/code/twin-diffusions/data/get_multi_view.py"


def renderViews(filepath, savepath, glb_filename):

    bpy.ops.object.select_all(action="SELECT")

    bpy.ops.object.delete()

    bpy.ops.import_scene.gltf(filepath=filepath, filter_glob="*.glb")
    print(filepath)

    for ob in bpy.context.scene.objects:
        # 如果ob.name里有数字
        if any(char.isdigit() for char in ob.name):
            ob.select_set(True)
            bpy.context.view_layer.objects.active = ob
    bpy.ops.object.join()

    obj = bpy.context.object

    bpy.context.view_layer.update()

    # 2.将模型移动到原点--------------------------------------
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS")

    obj.location = (0, 0, 0)

    # 3.计算缩放因子--------------------------------------
    # 计算模型的长宽高

    bbox = obj.bound_box
    width = (bbox[6][0] - bbox[0][0]) * obj.scale.x
    height = (bbox[6][1] - bbox[0][1]) * obj.scale.y
    depth = (bbox[6][2] - bbox[0][2]) * obj.scale.z

    # 计算缩放因子，使模型适应2x2x2的正方体
    scale_factor = 2 / max(width, height, depth)

    # 缩放模型
    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))

    # 5.更新场景--------------------------------------
    bpy.context.view_layer.update()

    # 6.渲染8个视角的视图-----------------------------
    # 定义相机位置和角度
    import math

    # Convert degrees to radians

    azimuths = [math.radians(deg) for deg in range(-180, 180, 45)]
    elevation = math.radians(30)

    viewpoints = []

    for azimuth in azimuths:
        # Calculate the camera coordinates
        x = 4 * math.cos(azimuth)
        y = 4 * math.sin(azimuth)
        z = 4 * math.tan(elevation)

        # Calculate the camera rotation
        rot_x = math.pi / 2 - elevation
        rot_y = 0
        rot_z = azimuth + math.pi / 2

        viewpoints.append({"location": (x, y, z), "rotation": (rot_x, rot_y, rot_z)})

    for i, viewpoint in enumerate(viewpoints):
        # Creating A New Camera Angle
        scene = bpy.context.scene
        cam = bpy.data.cameras.new("Camera")
        cam.lens = 50
        # create the second camera object
        cam_obj = bpy.data.objects.new("Camera", cam)
        # 设置相机位置和角度
        cam_obj.location = viewpoints[i]["location"]
        cam_obj.rotation_euler = viewpoints[i]["rotation"]
        scene.collection.objects.link(cam_obj)
        # Set the Camera to active camera
        bpy.context.scene.camera = bpy.data.objects["Camera"]

        # 为当前相机视角添加光源
        light_data = bpy.data.lights.new(name="New Light", type="POINT")
        light_data.energy = 50  # 设置光源强度
        light_object = bpy.data.objects.new(name="New Light", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        # 设置光源位置
        light_object.location = viewpoints[i]["location"]  # 设置光源的位置坐标

        # 保存渲染的图像
        FILE_PATH = os.path.join(
            savepath, f"{glb_filename}_{i}.png"
        )  # f"{obj.name}_{i}.png"
        # 设置渲染分辨率宽度和高度
        scene.render.resolution_x = 1024  # 设置宽度为1920像素
        scene.render.resolution_y = 1024  # 设置高度为1080像素
        bpy.context.scene.camera = cam_obj
        bpy.context.scene.render.filepath = FILE_PATH
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":

    input_dir = r"D:\blast furnace\code\twin-diffusions\data\objaverse_5k_prompt"
    output_dir = r"D:\blast furnace\code\twin-diffusions\data\multi_view"
    # list = ["449501e5ac704151b2f913649617dc7c", "6010d990c39b4f778f8d246af74b4dbe","460a5ca01e0e409cafaa5611e564f04b"]
    list = r"D:\blast furnace\code\twin-diffusions\data\test.txt"

    glb_list = []
    filename_list = []
    with open(
        list,
        "r",
    ) as file:

        for line in file:
            line = line.strip()
            filename = line
            filename_list.append(filename)
            glb_file = os.path.join(input_dir, line + ".glb")
            glb_list.append(glb_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Found {len(glb_list)} glb files")

    for filename in filename_list:
        input_file = os.path.join(input_dir, filename + ".glb")
        if not os.path.exists(input_file):
            print(f"File {input_file} not found")
            continue
        try:
            renderViews(input_file, output_dir, filename)
        except Exception as e:
            print(f"Error: {e}")
            continue
