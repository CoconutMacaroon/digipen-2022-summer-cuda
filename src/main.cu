#include "main.h"

__device__ double dot(double *x, double *y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

// function to add the elements of two arrays
__device__ void add(int n, double *x, double *y, double *z) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        z[i] = x[i] + y[i];
}

__device__ void add(double *a, double *b, double *resultLocation) {
    resultLocation[0] = a[0] + b[0];
    resultLocation[1] = a[1] + b[1];
    resultLocation[2] = a[2] + b[2];
}

__device__ void subtract(double *a, double *b, double *resultLocation) {
    resultLocation[0] = a[0] - b[0];
    resultLocation[1] = a[1] - b[1];
    resultLocation[2] = a[2] - b[2];
}


__device__ void multiply(double a, double *b, double *resultLocation) {
    resultLocation[0] = a * b[0];
    resultLocation[1] = a * b[1];
    resultLocation[2] = a * b[2];
}

__device__ void canvasToViewport(int x, int y, double *returnLocation) {
    returnLocation[0] = x * VIEWPORT_WIDTH / (double) CANVAS_WIDTH;
    returnLocation[1] = y * VIEWPORT_HEIGHT / (double) CANVAS_HEIGHT;
    returnLocation[2] = D;
}

__device__ void reflectRay(double R[], double N[], double *returnLocation) {
    double n_dot_r = dot(N, R);
    double n_multiply_two[3];
    multiply(2, N, n_multiply_two);

    double dot_times_multiply[3];
    multiply(n_dot_r, n_multiply_two, dot_times_multiply);

    subtract(dot_times_multiply, R, returnLocation);
}


__device__ void intersectRaySphere(double cameraPos[], double d[], Sphere sphere, double *returnLocation) {
    double CO[3];
    subtract(cameraPos, sphere.center, CO);

    double a = dot(d, d);
    double b = 2 * dot(CO, d);
    double c = dot(CO, CO) - sphere.radius * sphere.radius;

    double discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        returnLocation[0] = inf;
        returnLocation[1] = inf;
        return;
    }

    double discriminantSqrt = sqrt(discriminant);

    returnLocation[0] = (double) ((-b + discriminantSqrt) / (2 * a));
    returnLocation[1] = (double) ((-b - discriminantSqrt) / (2 * a));
}


__device__ IntersectionData closestIntersection(double cameraPos[], double d[], double t_min, double t_max) {
    double closest_t = DBL_MAX;
    Sphere closestSphere;
    bool isNull = true;
    for (size_t i = 0; i < ARR_LEN(spheres); ++i) {
        double t[2];
        intersectRaySphere(cameraPos, d, spheres[i], t);

        if (t[0] < closest_t && t_min < t[0] && t[0] < t_max) {
            closest_t = t[0];
            closestSphere = spheres[i];
            isNull = false;
        }
        if (t[1] < closest_t && t_min < t[1] && t[1] < t_max) {
            closest_t = t[1];
            closestSphere = spheres[i];
            isNull = false;
        }
    }
    IntersectionData data = {.sphere = closestSphere, .closest_t = closest_t, .isSphereNull = isNull};
    return data;
}


__device__ double computeLighting(double P[], double N[], double V[], double s) {
    double intensity = 0.0;
    for (size_t i = 0; i < ARR_LEN(lights); ++i) {
        if (lights[i].lightType == LIGHT_TYPE_AMBIENT) {
            intensity += lights[i].intensity;
        } else {
            double L[3];
            double t_max;
            if (lights[i].lightType == LIGHT_TYPE_POINT) {
                subtract(lights[i].position, P, L);
                t_max = 1.0;
            } else {
                L[0] = lights[i].direction[0];
                L[1] = lights[i].direction[1];
                L[2] = lights[i].direction[2];
                t_max = DBL_MAX;
            }
            // shadow check
            IntersectionData intersectionData = closestIntersection(P, L, 0.001f, t_max);

            if (!intersectionData.isSphereNull) {
                continue;
            }

            double Nf[] = {(double) N[0], (double) N[1], (double) N[2]};
            // diffuse
            double n_dot_l = dot(Nf, L);

            if (n_dot_l > 0) {
                intensity += lights[i].intensity * n_dot_l / (LENGTH(N) * LENGTH(L));
            }

            // specular
            if (s != -1) {
                // 2 * N * dot(N, L) - L
                double R[3];

                reflectRay(L, N, R);

                // double r_dot_v = dot(R, V);
                double r_dot_v = dot(R, V);

                if (r_dot_v > 0) {
                    intensity += lights[i].intensity * pow(r_dot_v / (LENGTH(R) * LENGTH(V)), s);
                }
            }
        }
    }
    return intensity;
}

__device__ byte roundRGBValue(double n) {
    return round(n) > 255.0 ? 255 : (byte) round(n);
}

__device__ Color traceRay(double cameraPos[3], double d[], double min_t, double max_t, int recursion_depth) {
    IntersectionData intersectionData = closestIntersection(cameraPos, d, min_t, max_t);
    if (intersectionData.isSphereNull) {
        // this is the background color
        return (Color) {0, 0, 0};
    }
    double tmp1[3];
    double d_double[] = {(double) d[0], (double) d[1], (double) d[2]};
    multiply(intersectionData.closest_t, d_double, tmp1);

    double P[3];
    add(cameraPos, tmp1, P);

    double N[3];
    subtract(P, intersectionData.sphere.center, N);

    double N2[3];
    multiply(1.0 / LENGTH(N), N, N2);

    double tmp3[3];
    multiply(-1.0, d, tmp3);

    double lighting = computeLighting(P, N, tmp3, intersectionData.sphere.specular);
    Color localColor = {roundRGBValue(intersectionData.sphere.color.r * lighting),
                        roundRGBValue(intersectionData.sphere.color.g * lighting),
                        roundRGBValue(intersectionData.sphere.color.b * lighting)};
    double r = intersectionData.sphere.reflectiveness;
    if (recursion_depth <= 0 || r <= 0) {
        return localColor;
    }

    double temp[3];
    multiply(-1.0, d, temp);
    double R[3];
    reflectRay(temp, N2, R);

    Color reflectedColor = traceRay(P, R, 0.001, inf, recursion_depth - 1);
    return (Color) {roundRGBValue(localColor.r * (1 - r) + reflectedColor.r * r),
                    roundRGBValue(localColor.g * (1 - r) + reflectedColor.g * r),
                    roundRGBValue(localColor.b * (1 - r) + reflectedColor.b * r)};
}

__device__ void putPixel(int x, int y, Color color) {
    /*x = CANVAS_WIDTH / 2 + x;
    y = CANVAS_HEIGHT / 2 - y - 1;*/
    if (x < 0 || x >= CANVAS_WIDTH || y < 0 || y >= CANVAS_HEIGHT) {
        return;
    }
    /*
    fputc((byte) x, image);
    fputc((byte) (x >> 8), image);

    fputc((byte) y, image);
    fputc((byte) (y >> 8), image);

    fputc(color.r, image);
    fputc(color.g, image);
    fputc(color.b, image);
    */
}

__device__ void renderPixel(int x, int y) {
    double d[3];
    canvasToViewport(x, y, d);
    // 4 is the number of reflections to calculate
    Color color = traceRay(cameraPosition, d, 1, inf, 4);
    putPixel(x, y, color);
}

typedef struct Pixel {
    short x, y;
} Pixel;

__global__ void doIt(Pixel *pixels) {
    const int n = CANVAS_WIDTH * CANVAS_HEIGHT;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        // printf("Rendering pixel (%d, %d)\n", pixels[i].x, pixels[i].y);
        renderPixel(pixels[i].x, pixels[i].y);
    }
}

int main(void) {
    Pixel *pixels = (Pixel *) malloc((size_t) CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(Pixel));

    size_t counter = 0;

    // TODO: in theory the index could be x * CANVAS_WIDTH + y
    // need to test that

    for (short x = 0; x < CANVAS_WIDTH; ++x) {
        for (short y = 0; y < CANVAS_HEIGHT; ++y) {
            pixels[counter++] = (Pixel) {.x = x, .y = y};
        }
    }


    // make a pointer for the copy of the pixels on
    // the GPU, and copy the pixels into that array
    Pixel *device_pixels;

    cudaMalloc(&device_pixels, CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(Pixel));

    cudaMemcpy(device_pixels, pixels, CANVAS_WIDTH * CANVAS_HEIGHT * sizeof(Pixel), cudaMemcpyHostToDevice);

    // run the kernel
    doIt<<<(CANVAS_WIDTH * CANVAS_HEIGHT + 256 - 1) / 256, 256>>>(device_pixels);
    cudaDeviceSynchronize();

    free(pixels);
    cudaFree(device_pixels);

    return EXIT_SUCCESS;
}
