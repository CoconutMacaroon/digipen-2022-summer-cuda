#include "main.h"

__device__ double dot(const double x[3], const double y[3]) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

__device__ void
add(const double a[], const double b[], double *resultLocation) {
    resultLocation[0] = a[0] + b[0];
    resultLocation[1] = a[1] + b[1];
    resultLocation[2] = a[2] + b[2];
}

__device__ void
subtract(const double a[], const double b[], double *resultLocation) {
    resultLocation[0] = a[0] - b[0];
    resultLocation[1] = a[1] - b[1];
    resultLocation[2] = a[2] - b[2];
}

__device__ void multiply(double a, const double b[], double *resultLocation) {
    resultLocation[0] = a * b[0];
    resultLocation[1] = a * b[1];
    resultLocation[2] = a * b[2];
}

__device__ void canvasToViewport(int x, int y, double *returnLocation) {
    returnLocation[0] = x * VIEWPORT_WIDTH / (double)CANVAS_WIDTH;
    returnLocation[1] = y * VIEWPORT_HEIGHT / (double)CANVAS_HEIGHT;
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

__device__ void intersectRaySphere(double cameraPos[],
                                   double d[],
                                   Sphere sphere,
                                   double *returnLocation) {
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

    returnLocation[0] = (double)((-b + discriminantSqrt) / (2 * a));
    returnLocation[1] = (double)((-b - discriminantSqrt) / (2 * a));
}

__device__ IntersectionData closestIntersection(double cameraPos[],
                                                double d[],
                                                double t_min,
                                                double t_max) {
    double closest_t = inf;
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
    IntersectionData data = {.sphere = closestSphere,
                             .closest_t = closest_t,
                             .isSphereNull = isNull};
    return data;
}

__device__ double
computeLighting(double P[], double N[], double V[], double s) {
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
            IntersectionData intersectionData =
                closestIntersection(P, L, 0.001f, t_max);

            if (!intersectionData.isSphereNull)
                continue;

            // diffuse
            double n_dot_l = dot(N, L);

            if (n_dot_l > 0)
                intensity +=
                    lights[i].intensity * n_dot_l / (LENGTH(N) * LENGTH(L));

            // specular
            if (s != -1) {
                // 2 * N * dot(N, L) - L
                double R[3];

                reflectRay(L, N, R);

                double r_dot_v = dot(R, V);

                if (r_dot_v > 0)
                    intensity += lights[i].intensity *
                                 pow(r_dot_v / (LENGTH(R) * LENGTH(V)), s);
            }
        }
    }
    return intensity;
}

__device__ Color traceRay(double cameraPos[3],
                          double d[],
                          double min_t,
                          double max_t,
                          int recursion_depth) {
    IntersectionData intersectionData =
        closestIntersection(cameraPos, d, min_t, max_t);
    if (intersectionData.isSphereNull)
        return BACKGROUND_COLOR;

    double tmp1[3];
    multiply(intersectionData.closest_t, d, tmp1);

    double P[3];
    add(cameraPos, tmp1, P);

    double N[3];
    subtract(P, intersectionData.sphere.center, N);

    double N2[3];
    multiply(1.0 / LENGTH(N), N, N2);

    double tmp3[3];
    multiply(-1.0, d, tmp3);
    double lighting =
        computeLighting(P, N, tmp3, intersectionData.sphere.specular);
    Color localColor = {
        ROUND_COLOR(intersectionData.sphere.color.r * lighting),
        ROUND_COLOR(intersectionData.sphere.color.g * lighting),
        ROUND_COLOR(intersectionData.sphere.color.b * lighting)};
    double r = intersectionData.sphere.reflectiveness;
    if (recursion_depth <= 0 || r <= 0)
        return localColor;

    double temp[3];
    multiply(-1.0, d, temp);
    double R[3];
    reflectRay(temp, N2, R);

    Color reflectedColor = traceRay(P, R, 0.001, inf, recursion_depth - 1);
    return (Color){ROUND_COLOR(localColor.r * (1 - r) + reflectedColor.r * r),
                   ROUND_COLOR(localColor.g * (1 - r) + reflectedColor.g * r),
                   ROUND_COLOR(localColor.b * (1 - r) + reflectedColor.b * r)};
}

__device__ void
putPixel(int x, int y, Color color, PixelRenderData *renderData) {
    x = CANVAS_WIDTH / 2 + x;
    y = CANVAS_HEIGHT / 2 - y - 1;
    // TODO: can we check this BEFORE we do the rendering so we
    // don't render if the result would get discarded anyways
    if (x < 0 || x >= CANVAS_WIDTH || y < 0 || y >= CANVAS_HEIGHT) {
        return;
    }

    // array mapping code from https://stackoverflow.com/a/2151141
    renderData[CANVAS_WIDTH * x + y].x = x;
    renderData[CANVAS_WIDTH * x + y].y = y;

    renderData[CANVAS_WIDTH * x + y].r = color.r;
    renderData[CANVAS_WIDTH * x + y].g = color.g;
    renderData[CANVAS_WIDTH * x + y].b = color.b;
}

__device__ void renderPixel(int x, int y, PixelRenderData *data) {
    double d[3];
    canvasToViewport(x, y, d);
    putPixel(
        x,
        y,
        traceRay(cameraPosition, d, 1, inf, RECURSION_DEPTH_FOR_REFLECTIONS),
        data);
}

__global__ void launch(Pixel *pixels,
                       PixelRenderData *renderData,
                       int canvasWidth,
                       int canvasHeight) {
    int counter = 0;
    for (short x = -canvasWidth; x < canvasWidth; ++x)
        for (short y = -canvasHeight; y < canvasHeight; ++y)
            pixels[counter++] = (Pixel){.x = x, .y = y};

    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < counter; i += stride)
        renderPixel(pixels[i].x, pixels[i].y, renderData);
}

int main() {
    // this contains the coords of all the pixels that need to be rendered
    Pixel *pixelsToRender;
    cudaMallocManaged(&pixelsToRender,
                      CANVAS_WIDTH * CANVAS_HEIGHT * 4 * sizeof(Pixel));

    // setup SDL
    SDL_Init(SDL_INIT_EVERYTHING);
    SDL_Window *window = SDL_CreateWindow("CUDA Raytracer",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          CANVAS_WIDTH,
                                          CANVAS_HEIGHT,
                                          SDL_WINDOW_SHOWN);
    SDL_Renderer *renderer =
        SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");

    SDL_Texture *texture = SDL_CreateTexture(renderer,
                                             SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             CANVAS_WIDTH,
                                             CANVAS_HEIGHT);
    std::vector<unsigned char> pixels(CANVAS_WIDTH * CANVAS_HEIGHT * 4, 0);

    bool useLockTexture = false;

    bool running = true;

    // create a pixel buffer for CUDA
    PixelRenderData *data;
    cudaMallocManaged(&data,
                      sizeof(PixelRenderData) * CANVAS_WIDTH * CANVAS_HEIGHT);

    launch<<<1, 256>>>(pixelsToRender, data, CANVAS_WIDTH, CANVAS_HEIGHT);
    cudaDeviceSynchronize();

    // fast SDL2 rendering code from https://stackoverflow.com/a/33312056
    while (running) {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
        SDL_RenderClear(renderer);

        // handle events
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if ((SDL_QUIT == ev.type) ||
                (SDL_KEYDOWN == ev.type &&
                 SDL_SCANCODE_ESCAPE == ev.key.keysym.scancode)) {
                running = false;
                break;
            }

            if (SDL_KEYDOWN == ev.type &&
                SDL_SCANCODE_L == ev.key.keysym.scancode) {
                useLockTexture = !useLockTexture;
            }
        }

        for (int i = 0; i < CANVAS_WIDTH * CANVAS_HEIGHT; i++) {
            const unsigned int offset =
                (CANVAS_WIDTH * data[i].y * 4) + data[i].x * 4;

            pixels[offset + 0] = data[i].b;
            pixels[offset + 1] = data[i].g;
            pixels[offset + 2] = data[i].r;
            pixels[offset + 3] = SDL_ALPHA_OPAQUE;

            SDL_SetRenderDrawColor(
                renderer, data[i].r, data[i].g, data[i].b, SDL_ALPHA_OPAQUE);
            SDL_RenderDrawPoint(renderer, data[i].x, data[i].y);
        }
        if (useLockTexture) {
            unsigned char *lockedPixels = nullptr;
            int pitch = 0;
            SDL_LockTexture(texture,
                            nullptr,
                            reinterpret_cast<void **>(&lockedPixels),
                            &pitch);
            std::copy_n(pixels.data(), pixels.size(), lockedPixels);
            SDL_UnlockTexture(texture);
        } else {
            SDL_UpdateTexture(
                texture, nullptr, pixels.data(), CANVAS_WIDTH * 4);
        }
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        break;
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    cudaFree(pixelsToRender);
    return EXIT_SUCCESS;
}
