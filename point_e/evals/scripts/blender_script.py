"""
要在Blender中运行的脚本，以将3D模型渲染为RGBAD图像。

示例用法
blender-b-P blender_script.py--\
--输入路径..//examples/example_data/corgi.ply\
--output_path render_out

为用于计算的渲染传递“--camera_pose z-circular-elevated”
CLIP R-精度结果。
输出目录将包括每个渲染视图的元数据json文件，以及用于渲染的全局元数据文件。
每个图像将另存为每个通道（rgbad）的16位PNG文件的集合，以及完整的视图的灰度渲染。

"""

"""
Script to run within Blender to render a 3D model as RGBAD images.

Example usage

    blender -b -P blender_script.py -- \
        --input_path ../../examples/example_data/corgi.ply \
        --output_path render_out

Pass `--camera_pose z-circular-elevated` for the rendering used to compute
CLIP R-Precision results.

The output directory will include metadata json files for each rendered view,
as well as a global metadata file for the render. Each image will be saved as
a collection of 16-bit PNG files for each channel (rgbad), as well as a full
grayscale render of the view.
"""

import argparse
import json
import math
import os
import random
import sys

import bpy
from mathutils import Vector
from mathutils.noise import random_unit_vector

MAX_DEPTH = 5.0
FORMAT_VERSION = 6
UNIFORM_LIGHT_DIRECTION = [0.09387503, -0.63953443, -0.7630093]


#用于清理场景中的所有对象
def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

#用于清除场景中的灯光对象。
def clear_lights():
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Light):
            obj.select_set(True)
    bpy.ops.object.delete()


#这个函数根据给定的文件路径导入3D模型。
def import_model(path):
    clear_scene()
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".obj":
        bpy.ops.import_scene.obj(filepath=path)
    elif ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".dae":
        bpy.ops.wm.collada_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    else:
        raise RuntimeError(f"unexpected extension: {ext}")

#生成场景中所有根对象（没有父对象）的迭代器。
def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

#用于计算场景中所有对象的边界框。
def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

#生成场景中所有网格对象的迭代器。
def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

#将场景中的所有对象缩放到单位立方体中心。
def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)

    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")

#创建一个新的相机对象，并将其添加到场景中。
def create_camera():
    # https://b3d.interplanety.org/en/how-to-create-camera-through-the-blender-python-api/
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object

#设置相机的方向和位置。
def set_camera(direction, camera_dist=2.0):
    camera_pos = -camera_dist * direction
    bpy.context.scene.camera.location = camera_pos

    # https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
    rot_quat = direction.to_track_quat("-Z", "Y")
    bpy.context.scene.camera.rotation_euler = rot_quat.to_euler()

    bpy.context.view_layer.update()

#随机化相机的方向。
def randomize_camera(camera_dist=2.0):
    direction = random_unit_vector()
    set_camera(direction, camera_dist=camera_dist)

#根据时间值旋转相机。
def pan_camera(time, axis="Z", camera_dist=2.0, elevation=-0.1):
    angle = time * math.pi * 2
    direction = [-math.cos(angle), -math.sin(angle), -elevation]
    assert axis in ["X", "Y", "Z"]
    if axis == "X":
        direction = [direction[2], *direction[:2]]
    elif axis == "Y":
        direction = [direction[0], -elevation, direction[1]]
    direction = Vector(direction).normalized()
    set_camera(direction, camera_dist=camera_dist)

#根据时间值设置相机的位置。
def place_camera(time, camera_pose_mode="random", camera_dist_min=2.0, camera_dist_max=2.0):
    camera_dist = random.uniform(camera_dist_min, camera_dist_max)
    if camera_pose_mode == "random":
        randomize_camera(camera_dist=camera_dist)
    elif camera_pose_mode == "z-circular":
        pan_camera(time, axis="Z", camera_dist=camera_dist)
    elif camera_pose_mode == "z-circular-elevated":
        pan_camera(time, axis="Z", camera_dist=camera_dist, elevation=0.2617993878)
    else:
        raise ValueError(f"Unknown camera pose mode: {camera_pose_mode}")

#创建一个新的光源对象，并将其添加到场景中。
def create_light(location, energy=1.0, angle=0.5 * math.pi / 180):
    # https://blender.stackexchange.com/questions/215624/how-to-create-a-light-with-the-python-api-in-blender-2-92
    light_data = bpy.data.lights.new(name="Light", type="SUN")
    light_data.energy = energy
    light_data.angle = angle
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)

    direction = -location
    rot_quat = direction.to_track_quat("-Z", "Y")
    light_object.rotation_euler = rot_quat.to_euler()
    bpy.context.view_layer.update()

    bpy.context.collection.objects.link(light_object)
    light_object.location = location

#创建一个新的随机光源。
def create_random_lights(count=4, distance=2.0, energy=1.5):
    clear_lights()
    for _ in range(count):
        create_light(random_unit_vector() * distance, energy=energy)

#创建一个新的相机光源。
def create_camera_light():
    clear_lights()
    create_light(bpy.context.scene.camera.location, energy=5.0)

#创建一个新的均匀光源。
def create_uniform_light(backend):
    clear_lights()
    # Random direction to decorrelate axis-aligned sides.
    pos = Vector(UNIFORM_LIGHT_DIRECTION)
    angle = 0.0092 if backend == "CYCLES" else math.pi
    create_light(pos, energy=5.0, angle=angle)
    create_light(-pos, energy=5.0, angle=angle)

#创建一个新的顶点颜色着色器。
def create_vertex_color_shaders():
    # By default, Blender will ignore vertex colors in both the
    # Eevee and Cycles backends, since these colors aren't
    # associated with a material.
    #
    # What we do here is create a simple material shader and link
    # the vertex color to the material color.
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, (bpy.types.Mesh)):
            continue

        if len(obj.data.materials):
            # We don't want to override any existing materials.
            continue

        color_keys = (obj.data.vertex_colors or {}).keys()
        if not len(color_keys):
            # Many objects will have no materials *or* vertex colors.
            continue

        mat = bpy.data.materials.new(name="VertexColored")
        mat.use_nodes = True

        # There should be a Principled BSDF by default.
        bsdf_node = None
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf_node = node
        assert bsdf_node is not None, "material has no Principled BSDF node to modify"

        socket_map = {}
        for input in bsdf_node.inputs:
            socket_map[input.name] = input

        # Make sure nothing lights the object except for the diffuse color.
        socket_map["Specular"].default_value = 0.0
        socket_map["Roughness"].default_value = 1.0

        v_color = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        v_color.layer_name = color_keys[0]

        mat.node_tree.links.new(v_color.outputs[0], socket_map["Base Color"])

        obj.data.materials.append(mat)

#创建一个新的默认材质。
def create_default_materials():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            if not len(obj.data.materials):
                mat = bpy.data.materials.new(name="DefaultMaterial")
                mat.use_nodes = True
                obj.data.materials.append(mat)

#获得场景中所有材质的集合。
def find_materials():
    all_materials = set()
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, (bpy.types.Mesh)):
            continue
        for mat in obj.data.materials:
            all_materials.add(mat)
    return all_materials

#获得节点的输入。
def get_socket_value(tree, socket):
    default = socket.default_value
    if not isinstance(default, float):
        default = list(default)
    for link in tree.links:
        if link.to_socket == socket:
            return (link.from_socket, default)
    return (None, default)

#清除节点的输入。
def clear_socket_input(tree, socket):
    for link in list(tree.links):
        if link.to_socket == socket:
            tree.links.remove(link)

#设置节点的输入。
def set_socket_value(tree, socket, socket_and_default):
    clear_socket_input(tree, socket)
    old_source_socket, default = socket_and_default
    if isinstance(default, float) and not isinstance(socket.default_value, float):
        # Codepath for setting Emission to a previous alpha value.
        socket.default_value = [default] * 3 + [1.0]
    else:
        socket.default_value = default
    if old_source_socket is not None:
        tree.links.new(old_source_socket, socket)

#设置Blender中的节点，通常用于渲染图像或数据。主要目的是根据输入的参数设置渲染节点树，最终输出渲染结果到文件。
def setup_nodes(output_path, capturing_material_alpha: bool = False):
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    # Helpers to perform math on links and constants. 
    #在链接和常量上执行数学运算的帮助函数。
    def node_op(op: str, *args, clamp=False):
        node = tree.nodes.new(type="CompositorNodeMath")
        node.operation = op
        if clamp:
            node.use_clamp = True
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)):
                node.inputs[i].default_value = arg
            else:
                links.new(arg, node.inputs[i])
        return node.outputs[0]

    def node_clamp(x, maximum=1.0):
        return node_op("MINIMUM", x, maximum)

    def node_mul(x, y, **kwargs):
        return node_op("MULTIPLY", x, y, **kwargs)

    input_node = tree.nodes.new(type="CompositorNodeRLayers")
    input_node.scene = bpy.context.scene

    input_sockets = {}
    for output in input_node.outputs:
        input_sockets[output.name] = output

    if capturing_material_alpha:
        color_socket = input_sockets["Image"]
    else:
        raw_color_socket = input_sockets["Image"]

        # We apply sRGB here so that our fixed-point depth map and material
        # alpha values are not sRGB, and so that we perform ambient+diffuse
        # lighting in linear RGB space.
        color_node = tree.nodes.new(type="CompositorNodeConvertColorSpace")
        color_node.from_color_space = "Linear"
        color_node.to_color_space = "sRGB"
        tree.links.new(raw_color_socket, color_node.inputs[0])
        color_socket = color_node.outputs[0]
    split_node = tree.nodes.new(type="CompositorNodeSepRGBA")
    tree.links.new(color_socket, split_node.inputs[0])
    # Create separate file output nodes for every channel we care about.
    # The process calling this script must decide how to recombine these
    # channels, possibly into a single image.
    for i, channel in enumerate("rgba") if not capturing_material_alpha else [(0, "MatAlpha")]:
        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        output_node.base_path = f"{output_path}_{channel}"
        links.new(split_node.outputs[i], output_node.inputs[0])

    if capturing_material_alpha:
        # No need to re-write depth here.
        return

    depth_out = node_clamp(node_mul(input_sockets["Depth"], 1 / MAX_DEPTH))
    output_node = tree.nodes.new(type="CompositorNodeOutputFile")
    output_node.base_path = f"{output_path}_depth"
    links.new(depth_out, output_node.inputs[0])

#用于渲染场景的函数，它会渲染出不同通道（RGBA 和深度）的图像，并将结果输出到指定路径。
def render_scene(output_path, fast_mode: bool):
    use_workbench = bpy.context.scene.render.engine == "BLENDER_WORKBENCH"
    if use_workbench:
        # We must use a different engine to compute depth maps.
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
        bpy.context.scene.eevee.taa_render_samples = 1  # faster, since we discard image.
    if fast_mode:
        if bpy.context.scene.render.engine == "BLENDER_EEVEE":
            bpy.context.scene.eevee.taa_render_samples = 1
        elif bpy.context.scene.render.engine == "CYCLES":
            bpy.context.scene.cycles.samples = 256
    else:
        if bpy.context.scene.render.engine == "CYCLES":
            # We should still impose a per-frame time limit
            # so that we don't timeout completely.
            bpy.context.scene.cycles.time_limit = 40
    bpy.context.view_layer.update()
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.view_settings.view_transform = "Raw"  # sRGB done in graph nodes
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "BW"
    bpy.context.scene.render.image_settings.color_depth = "16"
    bpy.context.scene.render.filepath = output_path
    setup_nodes(output_path)
    bpy.ops.render.render(write_still=True)

    # The output images must be moved from their own sub-directories, or
    # discarded if we are using workbench for the color.
    for channel_name in ["r", "g", "b", "a", "depth"]:
        sub_dir = f"{output_path}_{channel_name}"
        image_path = os.path.join(sub_dir, os.listdir(sub_dir)[0])
        name, ext = os.path.splitext(output_path)
        if channel_name == "depth" or not use_workbench:
            os.rename(image_path, f"{name}_{channel_name}{ext}")
        else:
            os.remove(image_path)
        os.removedirs(sub_dir)

    if use_workbench:
        # Re-render RGBA using workbench with texture mode, since this seems
        # to show the most reasonable colors when lighting is broken.
        bpy.context.scene.use_nodes = False
        bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.image_settings.color_depth = "8"
        bpy.context.scene.display.shading.color_type = "TEXTURE"
        bpy.context.scene.display.shading.light = "FLAT"
        if fast_mode:
            # Single pass anti-aliasing.
            bpy.context.scene.display.render_aa = "FXAA"
        os.remove(output_path)
        bpy.ops.render.render(write_still=True)
        bpy.context.scene.render.image_settings.color_mode = "BW"
        bpy.context.scene.render.image_settings.color_depth = "16"

#计算场景中相机的视场角（Field of View，FOV）
def scene_fov():
    x_fov = bpy.context.scene.camera.data.angle_x
    y_fov = bpy.context.scene.camera.data.angle_y
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    if bpy.context.scene.camera.data.angle == x_fov:
        y_fov = 2 * math.atan(math.tan(x_fov / 2) * height / width)
    else:
        x_fov = 2 * math.atan(math.tan(y_fov / 2) * width / height)
    return x_fov, y_fov

#将相机的元数据写入到指定路径的文件中。
def write_camera_metadata(path):
    x_fov, y_fov = scene_fov()
    bbox_min, bbox_max = scene_bbox()
    matrix = bpy.context.scene.camera.matrix_world
    with open(path, "w") as f:
        json.dump(
            dict(
                format_version=FORMAT_VERSION,
                max_depth=MAX_DEPTH,
                bbox=[list(bbox_min), list(bbox_max)],
                origin=list(matrix.col[3])[:3],
                x_fov=x_fov,
                y_fov=y_fov,
                x=list(matrix.col[0])[:3],
                y=list(-matrix.col[1])[:3],
                z=list(-matrix.col[2])[:3],
            ),
            f,
        )

#用于生成渲染数据集，并保存到指定路径中。
def save_rendering_dataset(
    input_path: str,
    output_path: str,
    num_images: int,
    backend: str,
    light_mode: str,
    camera_pose: str,
    camera_dist_min: float,
    camera_dist_max: float,
    fast_mode: bool,
):
    assert light_mode in ["random", "uniform", "camera"]
    assert camera_pose in ["random", "z-circular", "z-circular-elevated"]

    import_model(input_path)
    bpy.context.scene.render.engine = backend
    normalize_scene()
    if light_mode == "random":
        create_random_lights()
    elif light_mode == "uniform":
        create_uniform_light(backend)
    create_camera()
    create_vertex_color_shaders()
    for i in range(num_images):
        t = i / max(num_images - 1, 1)  # same as np.linspace(0, 1, num_images)
        place_camera(
            t,
            camera_pose_mode=camera_pose,
            camera_dist_min=camera_dist_min,
            camera_dist_max=camera_dist_max,
        )
        if light_mode == "camera":
            create_camera_light()
        render_scene(
            os.path.join(output_path, f"{i:05}.png"),
            fast_mode=fast_mode,
        )
        write_camera_metadata(os.path.join(output_path, f"{i:05}.json"))
    with open(os.path.join(output_path, "info.json"), "w") as f:
        info = dict(
            backend=backend,
            light_mode=light_mode,
            fast_mode=fast_mode,
            format_version=FORMAT_VERSION,
            channels=["R", "G", "B", "A", "D"],
            scale=0.5,  # The scene is bounded by [-scale, scale].
        )
        json.dump(info, f)


def main():
    try:
        dash_index = sys.argv.index("--")
    except ValueError as exc:
        raise ValueError("arguments must be preceded by '--'") from exc

    raw_args = sys.argv[dash_index + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--backend", type=str, default="BLENDER_EEVEE")
    parser.add_argument("--light_mode", type=str, default="uniform")
    parser.add_argument("--camera_pose", type=str, default="random")
    parser.add_argument("--camera_dist_min", type=float, default=2.0)
    parser.add_argument("--camera_dist_max", type=float, default=2.0)
    parser.add_argument("--fast_mode", action="store_true")
    args = parser.parse_args(raw_args)

    save_rendering_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        num_images=args.num_images,
        backend=args.backend,
        light_mode=args.light_mode,
        camera_pose=args.camera_pose,
        camera_dist_min=args.camera_dist_min,
        camera_dist_max=args.camera_dist_max,
        fast_mode=args.fast_mode,
    )


main()
