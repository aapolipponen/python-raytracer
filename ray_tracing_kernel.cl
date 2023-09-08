#define NUM_SPHERES 3

typedef struct {
    float3 center;
    float radius;
    float3 color;
} Sphere;

bool sphere_intersection(float3 ray_origin, float3 ray_direction, Sphere sphere, float *t_result) {
    float3 oc = ray_origin - sphere.center;
    float a = dot(ray_direction, ray_direction);
    float b = 2.0f * dot(oc, ray_direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        return false;
    } else {
        float t1 = (-b - sqrt(discriminant)) / (2.0f * a);
        float t2 = (-b + sqrt(discriminant)) / (2.0f * a);
        *t_result = t1 < t2 ? t1 : t2;
        return true;
    }
}

float3 sphere_normal(float3 intersection_point, float3 sphere_center) {
    return normalize(intersection_point - sphere_center);
}

__kernel void ray_trace(
    __global uchar4 *output,
    const unsigned int width,
    const unsigned int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float aspect_ratio = (float)width / (float)height;
    float3 ray_origin = (float3)(0.0f, 0.0f, 0.0f);
    float3 ray_direction = (float3)(
        (x - width / 2.0f) / (width / 2.0f) * aspect_ratio,
        (y - height / 2.0f) / (height / 2.0f),
        1.0f
    );

    ray_direction = normalize(ray_direction);

    Sphere spheres[NUM_SPHERES];
    spheres[0] = (Sphere){{0.0f, -1.0f, 5.0f}, 1.0f, {1.0f, 0.0f, 0.0f}}; // Red
    spheres[1] = (Sphere){{-2.0f, -1.0f, 4.0f}, 0.5f, {0.0f, 1.0f, 0.0f}}; // Green
    spheres[2] = (Sphere){{2.0f, -0.5f, 6.0f}, 0.5f, {0.0f, 0.0f, 1.0f}}; // Blue

    // Light source
    float3 light_position = (float3)(0.0f, 5.0f, 0.0f);

    uchar4 pixel_color = (uchar4)(0, 0, 0, 255); // Default: Black background

    for (int i = 0; i < NUM_SPHERES; i++) {
        float t_intersection;
        if(sphere_intersection(ray_origin, ray_direction, spheres[i], &t_intersection)) {
            float3 intersection_point = ray_origin + t_intersection * ray_direction;
            float3 normal = sphere_normal(intersection_point, spheres[i].center);
            float3 to_light = normalize(light_position - intersection_point);
            
            float ambient = 0.1f;
            float diffuse = max(0.0f, dot(normal, to_light));

            // Shadow check
            bool in_shadow = false;
            for (int j = 0; j < NUM_SPHERES; j++) {
                if (i != j) {
                    float t_shadow;
                    if(sphere_intersection(intersection_point + 0.001f * to_light, to_light, spheres[j], &t_shadow)) {
                        float distance_to_light = length(light_position - intersection_point);
                        if(t_shadow < distance_to_light) {
                            in_shadow = true;
                            break;
                        }
                    }
                }
            }

            if(in_shadow) {
                diffuse *= 0.5; // Half the diffuse light if in shadow
            }

            float3 color = spheres[i].color * (ambient + diffuse);
            pixel_color = (uchar4)(clamp(color.x * 255.0f, 0.0f, 255.0f), 
                                   clamp(color.y * 255.0f, 0.0f, 255.0f),
                                   clamp(color.z * 255.0f, 0.0f, 255.0f), 255);
            break;
        }
    }

    output[y * width + x] = pixel_color;
}