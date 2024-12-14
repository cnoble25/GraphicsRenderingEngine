#include "color.h"
#include "vec3.h"
#include "ray.h"
#include <vector>
#include <cmath>
#include <iostream>
#include "vertex.h"
#include "model.h"
#include "transform.h"
#include "rotation.h"

color ray_color(const ray& r, const double luminosity = 1.0) {
    auto first_point = point3(4,-1, 5);
    auto second_point = point3(5,-4,5);
    auto third_point = point3(0,-1,5);
    // vertex v = vertex(first_point, second_point, third_point, color(0,0,0));
    // vertex u = vertex(first_point, second_point, third_point, color(0,0,0));
    // model m = model({v,u}, transforms(vec3(0,0,10), rotations(0, 0, 0), vec3(1,1,1)));
    model m = pyamid();
    double t = m.intersect(r);
    if(t > -0.5) {
        return (luminosity/(t))*color(1,1,1);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * unit_direction.y()+1.0;
    return (1.0-a)*color(1,1,1) + a*color(0.5, 0.7, 1.0);
}

int main() {

    auto aspect_ratio = 16.0/9.0;
    int image_width = 400;

    int image_height = int(image_width/aspect_ratio);
    image_height = (image_height < 1) ? 1: image_height;

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(aspect_ratio));
    auto camera_center = point3(0,0,0);

    auto viewport_u = vec3(viewport_width,0,0);
    auto viewport_v = vec3(0,-viewport_height,0);

    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;

    auto pixel100_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);


    // Render

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = image_height; j > 0; j--) {
        for (int i = 0; i < image_width; i++) {
            auto pixel_center = pixel100_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto ray_direction = camera_center-pixel_center;
            ray r(pixel_center, ray_direction);

            color pixel_color = ray_color(r, 5);
            write_color(std::cout, pixel_color);

        }
    }
}