import numpy as np
import concurrent.futures
import pyopencl as cl
from PIL import Image

platform = cl.get_platforms()[0]  
device = platform.get_devices()[0]  
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Global constants
WIDTH, HEIGHT = 700, 700  # Output image dimensions
MAX_RECURSION_DEPTH = 5
SAMPLES_PER_PIXEL = 5
TILE_SIZE = 32

class Vector3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __itruediv__(self, scalar):
        self.x /= scalar
        self.y /= scalar
        self.z /= scalar
        return self

    def __mul__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, (int, float)):
            return Vector3(self.x * other, self.y * other, self.z * other)
        else:
            raise ValueError("Unsupported type for multiplication")

    __rmul__ = __mul__  # This allows for reverse multiplication

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def magnitude(self):
        return np.sqrt(self.dot(self))

    def normalized(self):
        mag = self.magnitude()
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

class AreaLight:
    def __init__(self, position, size):
        self.position = position  # This can be the center of the area light
        self.size = size  # For simplicity, let's assume it's a square light source for now

    def sample_point(self):
        """Returns a random point on the light source."""
        half_size = self.size / 2
        dx = np.random.uniform(-half_size, half_size)
        dy = np.random.uniform(-half_size, half_size)
        return self.position + dx * Vector3(1, 0, 0) + dy * Vector3(0, 1, 0)

class Ray:
    def __init__(self, origin, direction):
        self.origin, self.direction = origin, direction

class Camera:
    def __init__(self, position, look_at, up, fov, aspect_ratio=1.0):
        self.position = position
        self.forward = (look_at - position).normalized()
        self.right = self.forward.cross(up).normalized()
        self.up = self.right.cross(self.forward).normalized()

        # Compute half dimensions using FOV
        self.half_height = np.tan(fov / 2.0)
        self.half_width = self.half_height * aspect_ratio
                                           
    def get_ray(self, u, v):
        direction = (self.forward 
                     - (u - 0.5) * 2.0 * self.half_width * self.right 
                     - (v - 0.5) * 2.0 * self.half_height * self.up)
        return Ray(self.position, direction.normalized())
            
class Sphere:
    def __init__(self, center, radius, color):
        self.center, self.radius, self.color = center, radius, color

    def intersect(self, ray):
        l = self.center - ray.origin
        tca = l.dot(ray.direction)
        d2 = l.dot(l) - tca * tca
        if d2 > self.radius * self.radius:
            return None
        thc = np.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 < 0:
            t0 = t1
        if t0 < 0:
            return None
        return t0

class Scene:
    def __init__(self):
        self.spheres = []
        self.lights = []
        self.camera = None  # Initially no camera

    def add_sphere(self, sphere):
        self.spheres.append(sphere)

    def add_light(self, light_position):
        self.lights.append(light_position)

    def set_camera(self, camera):
        self.camera = camera

def reflect(in_direction, normal):
    """Compute the reflection direction."""
    dot_product = in_direction.dot(normal)
    return in_direction - normal * 2 * dot_product

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def trace_ray(ray, scene, depth=0):
    if depth > MAX_RECURSION_DEPTH:
        return Vector3(0, 0, 0)  # Background color or termination color

    closest_sphere = None
    closest_t = float("inf")
    for sphere in scene.spheres:
        t = sphere.intersect(ray)
        if t and t < closest_t:
            closest_t = t
            closest_sphere = sphere

    if not closest_sphere:
        return Vector3(0, 0, 0)  # Background color

    hit_point = ray.origin + ray.direction * closest_t
    normal = (hit_point - closest_sphere.center).normalized()

    total_color = Vector3(0, 0, 0)
    
    for light in scene.lights:
        num_shadow_rays = 40  # For example
        in_shadow_count = 0

        for _ in range(num_shadow_rays):
            light_sample = light.sample_point()
            light_dir = (light_sample - hit_point).normalized()

            shadow_ray = Ray(hit_point + normal * 0.001, light_dir)  # Offset to avoid self-shadowing

            # Check if the shadow ray hits any object before reaching the light
            is_blocked = False
            for sphere in scene.spheres:
                t = sphere.intersect(shadow_ray)
                if t and t < (light_sample - hit_point).magnitude():
                    is_blocked = True
                    break
            in_shadow_count += is_blocked

        in_shadow_fraction = in_shadow_count / num_shadow_rays

        # Compute lighting, but attenuate based on the shadow fraction
        intensity = max(0, normal.dot(light_dir)) * (1 - in_shadow_fraction)
        total_color += closest_sphere.color * intensity

    # Handle reflection
    reflected_dir = reflect(ray.direction, normal).normalized()
    reflected_ray = Ray(hit_point + reflected_dir * 0.001, reflected_dir)  # 0.001 offset to prevent self-intersection
    reflection_color = trace_ray(reflected_ray, scene, depth + 1)
    total_color = total_color * 0.5 + reflection_color * 0.5  # 50% blend for this example

    return total_color

def generate_tiles(width, height, tile_size):
    """Yields (start_x, start_y, end_x, end_y) for each tile."""
    for x in range(0, width, tile_size):
        for y in range(0, height, tile_size):
            yield (x, y, min(x + tile_size, width), min(y + tile_size, height))

kernel_code = f"""
#define TILE_SIZE {TILE_SIZE}

__constant float3 BACKGROUND_COLOR = (float3)(0.5f, 0.7f, 0.9f);

typedef struct {{
    float3 center;
    float radius;
}} Sphere;

typedef struct {{
    float3 origin;
    float3 direction;
}} Ray;

float3 subtract(float3 a, float3 b) {{
    return (float3)(a.x - b.x, a.y - b.y, a.z - b.z);
}}

float my_dot(float3 a, float3 b) {{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}}

bool hit_sphere(Sphere sphere, Ray ray, float* distance) {{
    float3 oc = subtract(ray.origin, sphere.center);
    float a = my_dot(ray.direction, ray.direction);
    float b = 2.0f * my_dot(oc, ray.direction);
    float c = my_dot(oc, oc) - sphere.radius*sphere.radius;
    float discriminant = b*b - 4*a*c;
    if (discriminant > 0) {{
        *distance = (-b - sqrt(discriminant)) / (2.0f * a);
        return true;
    }}
    return false;
}}

__kernel void render_tile_kernel(__global float3* pixels,
                                 __global Sphere* spheres,
                                 __global float3* colors,
                                 int num_spheres,
                                 __constant float3* bg_color)
{{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);

    // Sample screen point (simplified for example)
    float3 lower_left_corner = (float3)(-2.0f, -1.0f, -1.0f);
    float3 horizontal = (float3)(4.0f, 0.0f, 0.0f);
    float3 vertical = (float3)(0.0f, 2.0f, 0.0f);
    float3 origin = (float3)(0.0f, 0.0f, 0.0f);

    float u = (float)gid_x / (float)TILE_SIZE;
    float v = (float)gid_y / (float)TILE_SIZE;

    float3 direction = lower_left_corner + u*horizontal + v*vertical - origin;
    Ray ray = (Ray) {{origin, direction}};

    float3 color = BACKGROUND_COLOR;

    for(int i = 0; i < num_spheres; i++) {{
        float distance;
        if(hit_sphere(spheres[i], ray, &distance)) {{
            color = (float3)(1.0f, 0.0f, 0.0f);  // Red for simplicity
            break;
        }}
    }}

    pixels[gid_y * TILE_SIZE + gid_x] = color;
}}
"""

def render_tile(start_x, start_y, end_x, end_y, sphere_data_np, sphere_colors_np, bg_color_buffer):
    local_size = (TILE_SIZE, TILE_SIZE)
    global_size = (end_x - start_x, end_y - start_y)

    pixels_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, global_size[0] * global_size[1] * 3 * 4)

    program.render_tile_kernel(queue, global_size, local_size, pixels_buffer, sphere_buffer, colors_buffer, np.int32(len(sphere_data_np)), bg_color_buffer)

    pixels = np.empty(global_size + (3,), dtype=np.float32)
    cl.enqueue_copy(queue, pixels, pixels_buffer)

    return start_x, start_y, pixels
    
def render_scene_concurrently(scene, bg_color_buffer):
    # Prepare scene data for the OpenCL kernels
    sphere_data = [(sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius) for sphere in scene.spheres]
    sphere_colors = [(sphere.color.x, sphere.color.y, sphere.color.z) for sphere in scene.spheres]

    # Convert data to numpy arrays
    sphere_data_np = np.array(sphere_data, dtype=np.float32)
    sphere_colors_np = np.array(sphere_colors, dtype=np.float32)

    # Create OpenCL buffers
    sphere_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=sphere_data_np.nbytes)
    colors_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=sphere_colors_np.nbytes)

    # Copy data to GPU
    cl.enqueue_copy(queue, sphere_buffer, sphere_data_np)
    cl.enqueue_copy(queue, colors_buffer, sphere_colors_np)

    # Create an empty array for the whole image
    image = np.empty((WIDTH, HEIGHT, 3), dtype=np.float32)

    # Use ThreadPoolExecutor to run the tile rendering concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for start_x, start_y, end_x, end_y in generate_tiles(WIDTH, HEIGHT, TILE_SIZE):
            futures.append(executor.submit(render_tile, start_x, start_y, end_x, end_y, sphere_data_np, sphere_colors_np, bg_color_buffer))

        for future in concurrent.futures.as_completed(futures):
            start_x, start_y, pixels = future.result()
            actual_tile_height = min(pixels.shape[0], image.shape[0] - start_y)
            actual_tile_width = min(pixels.shape[1], image.shape[1] - start_x)

            image[start_y:start_y + actual_tile_height, start_x:start_x + actual_tile_width] = pixels[:actual_tile_height, :actual_tile_width]
            
    return image


def save_image(pixels, filename="output.png"):
    # Replace NaN values with 0
    pixels[np.isnan(pixels)] = 0
    
    # Convert the pixels array to an unsigned 8-bit integer array
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    # Create an image from the pixels array
    image = Image.fromarray(pixels)

    # Save the image
    image.save(filename)

if __name__ == "__main__":
    scene = Scene()
    
    # Adding multiple spheres to the scene
    scene.add_sphere(Sphere(Vector3(0, 0, 5), 1, Vector3(255, 0, 0)))
    scene.add_sphere(Sphere(Vector3(2, 1, 8), 1.5, Vector3(0, 0, 255)))
    scene.add_sphere(Sphere(Vector3(-2, 1, 7), 1.2, Vector3(0, 255, 0)))
    # A large sphere as the "floor"
    scene.add_sphere(Sphere(Vector3(0, -4, 10), 3.5, Vector3(150, 150, 150)))

    # Adding multiple light sources
    scene.add_light(AreaLight(Vector3(2, 5, 0), 2))
    scene.add_light(AreaLight(Vector3(-5, 5, -5), 2))

    # Define the camera
    cam_position = Vector3(0, 0, 0)
    cam_look_at = Vector3(0, 0, 1)
    cam_up = Vector3(0, 1, 0)
    aspect_ratio = WIDTH / HEIGHT
    fov_in_degrees = 90
    fov_in_radians = np.radians(fov_in_degrees)
    camera = Camera(cam_position, cam_look_at, cam_up, fov_in_radians, aspect_ratio)
    scene.camera = camera

    sphere_data = [(sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius) for sphere in scene.spheres]
    sphere_colors = [(sphere.color.x, sphere.color.y, sphere.color.z) for sphere in scene.spheres]

    dummy_bg_color = np.array([0.5, 0.7, 0.9], dtype=np.float32)
    bg_color_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=dummy_bg_color.nbytes)

    # Convert data to numpy arrays
    sphere_data_np = np.array(sphere_data, dtype=np.float32)
    sphere_colors_np = np.array(sphere_colors, dtype=np.float32)

    # Create OpenCL buffers
    sphere_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=sphere_data_np.nbytes)
    colors_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=sphere_colors_np.nbytes)

    # Copy data to GPU
    cl.enqueue_copy(queue, sphere_buffer, sphere_data_np)
    cl.enqueue_copy(queue, colors_buffer, sphere_colors_np)
    cl.enqueue_copy(queue, bg_color_buffer, dummy_bg_color)

    # Prepare an empty buffer for the output pixels
    pixels_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, (TILE_SIZE * TILE_SIZE * 3 * 4))

    # Compile and run the kernel
    program = cl.Program(context, kernel_code).build()

    # Copy the result back
    pixels = np.empty((TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
    cl.enqueue_copy(queue, pixels, pixels_buffer)

    max_work_group_size = queue.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

    TILE_SIZE = int(np.sqrt(max_work_group_size))

    image = render_scene_concurrently(scene, bg_color_buffer)
    save_image(image)