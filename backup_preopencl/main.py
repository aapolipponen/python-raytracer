import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor


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

def render_tile(scene, start_x, start_y, end_x, end_y):
    pixels = [[Vector3(0, 0, 0) for _ in range(end_x - start_x)] for _ in range(end_y - start_y)]
    
    for i in range(start_y, end_y):
        for j in range(start_x, end_x):
            sample_colors = []

            for x in range(SAMPLES_PER_PIXEL):
                for y in range(SAMPLES_PER_PIXEL):
                    u_offset = (x + 0.5) / SAMPLES_PER_PIXEL
                    v_offset = (y + 0.5) / SAMPLES_PER_PIXEL

                    u = (j + u_offset) / WIDTH
                    v = (i + v_offset) / HEIGHT

                    ray = scene.camera.get_ray(u, v)
                    sample_colors.append(trace_ray(ray, scene))

            avg_color = sum(sample_colors, Vector3(0, 0, 0)) * (1.0 / (SAMPLES_PER_PIXEL ** 2))
            pixels[i - start_y][j - start_x] = avg_color

    return start_x, start_y, pixels

def save_image(pixels, filename="output.ppm"):
    with open(filename, "w") as f:
        f.write(f"P3\n{WIDTH} {HEIGHT}\n255\n")
        for row in pixels:
            for color in row:
                r, g, b = color.x, color.y, color.z
                # Ensure the color values are integers and clamped between 0 and 255
                r = int(np.clip(r, 0, 255))
                g = int(np.clip(g, 0, 255))
                b = int(np.clip(b, 0, 255))
                f.write(f"{r} {g} {b} ")
            f.write("\n")

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

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(render_tile, scene, x0, y0, x1, y1) for x0, y0, x1, y1 in generate_tiles(WIDTH, HEIGHT, TILE_SIZE)]
        
        pixels = [[Vector3(0, 0, 0) for _ in range(WIDTH)] for _ in range(HEIGHT)]
        
        for future in concurrent.futures.as_completed(futures):
            x0, y0, tile_pixels = future.result()
            for y, row in enumerate(tile_pixels):
                for x, color in enumerate(row):
                    pixels[y0 + y][x0 + x] = color

    save_image(pixels)