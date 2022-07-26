#include <iostream>
#include <float.h>
#include <stdlib.h>

const short LIGHT_TYPE_AMBIENT = 1;
const short LIGHT_TYPE_POINT = 2;
const short LIGHT_TYPE_DIRECTIONAL = 3;

#define LENGTH(n) (sqrt(dot(n, n)))

#define ARR_LEN(a) (sizeof(a) / sizeof(a[0]))

typedef unsigned char byte;

typedef struct Color {
    int r, g, b;
} Color;

typedef struct PixelColorData {
    short x, y;
    byte r, g, b;
} PixelColorData;

typedef struct Sphere {
    double radius;
    double center[3];
    Color color;
    double specular;
    double reflectiveness;
} Sphere;

typedef struct IntersectionData {
    Sphere sphere;
    double closest_t;
    bool isSphereNull;
} IntersectionData;

typedef struct Light {
    short lightType;
    double intensity;
    double position[3];
    double direction[3];
} Light;

__device__ const short CANVAS_WIDTH = 1024, CANVAS_HEIGHT = 1024;

__device__ double D = 1;
// TODO: I may need to swap CANVAS_WIDTH and CANVAS_HEIGHT
//       in this division if CANVAS_HEIGHT > CANVAS_WIDTH
__device__ double VIEWPORT_WIDTH = (double) CANVAS_WIDTH / (double) CANVAS_HEIGHT;
__device__ double VIEWPORT_HEIGHT = 1;
__device__ double inf = DBL_MAX;
__device__ double cameraPosition[] = {0, 0, 0};

__device__ Sphere
spheres[] = {
{
.
radius = 1.0,
.
center = {-2, 0, 4},
.
color = {0, 255,
         0},
.
specular = 500,
.
reflectiveness = 0.2
},
{
.
radius = 1.0,
.
center = {2, 0, 4},
.
color = {0, 0,
         255},
.
specular = 500,
.
reflectiveness = 0.3
},
{
.
radius = 1.0,
.
center = {0, -1, 3},
.
color = {255, 0,
         0},
.
specular = 10,
.
reflectiveness = 0.4
},
{
.
radius = 5000,
.
center = {0, -5001, 0},
.
color = {255, 255,
         0},
.
specular = 1000,
.
reflectiveness = 0.5
}};


__device__ Light
lights[] = {
(Light) {
.
lightType = LIGHT_TYPE_AMBIENT,
.
intensity = 0.2,
.
position = {},
.
direction = {}
},
(Light) {
.
lightType = LIGHT_TYPE_POINT,
.
intensity = 0.6,
.
position = {2.0, 1.0,
            0.0},
.
direction = {}
},
(Light) {
.
lightType = LIGHT_TYPE_DIRECTIONAL,
.
intensity = 0.2f,
.
position = {},
.
direction = {1.0,
             4.0,
             4.0}
}};
