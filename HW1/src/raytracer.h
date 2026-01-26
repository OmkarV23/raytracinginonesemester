#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <cmath>
#include <iostream>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define HYBRID_FUNC __host__ __device__
    using Vec3 = float3; // Use CUDA's float3
#else
    #define HYBRID_FUNC
    struct float3_cpu { float x, y, z; };
    using Vec3 = float3_cpu; // Use custom CPU vector
#endif


HYBRID_FUNC inline Vec3 make_vec3(float x, float y, float z) {
    Vec3 v; v.x = x; v.y = y; v.z = z; return v;
}
HYBRID_FUNC inline Vec3 operator+(const Vec3& a, const Vec3& b) { return make_vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
HYBRID_FUNC inline Vec3 operator-(const Vec3& a, const Vec3& b) { return make_vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
HYBRID_FUNC inline Vec3 operator*(const Vec3& a, const Vec3& b) { return make_vec3(a.x*b.x, a.y*b.y, a.z*b.z); }
HYBRID_FUNC inline Vec3 operator*(const Vec3& v, float t)       { return make_vec3(v.x*t, v.y*t, v.z*t); }
HYBRID_FUNC inline Vec3 operator*(float t, const Vec3& v)       { return make_vec3(v.x*t, v.y*t, v.z*t); }
HYBRID_FUNC inline float dot(const Vec3& u, const Vec3& v)      { return u.x*v.x + u.y*v.y + u.z*v.z; }
HYBRID_FUNC inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return make_vec3(u.y * v.z - u.z * v.y,
                     u.z * v.x - u.x * v.z,
                     u.x * v.y - u.y * v.x);
}

HYBRID_FUNC inline Vec3 unit_vector(Vec3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return make_vec3(v.x/len, v.y/len, v.z/len);
}


enum MaterialType { MAT_LAMBERTIAN, MAT_METAL };

struct Material {
    MaterialType type;
    Vec3 albedo;
    float fuzz;
    float shininess;
};

struct HitRecord {
    bool hit;
    float t;
    Vec3 p;
    Vec3 normal;
    Material mat;
};

struct Ray {
    Vec3 origin;
    Vec3 dir;
    HYBRID_FUNC Vec3 at(float t) const { return origin + dir * t; }
};

struct Light {
    Vec3 position;
    Vec3 color;
};


HYBRID_FUNC HitRecord world_intersect(const Ray& r) {
    HitRecord rec; rec.hit = false; rec.t = 1e30f;
    
    // Hardcoded Sphere: (0,0,-1) Radius 0.5
    Vec3 center = make_vec3(0, 0, -1);
    float radius = 0.5f;
    Vec3 oc = r.origin - center;
    float a = dot(r.dir, r.dir);
    float b = 2.0f * dot(oc, r.dir);
    float c = dot(oc, oc) - radius*radius;
    float disc = b*b - 4*a*c;

    if (disc > 0) {
        float temp = (-b - sqrtf(disc)) / (2.0f*a);
        if (temp > 0.001f) {
            rec.hit = true;
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = unit_vector(rec.p - center);
            rec.mat.type = MAT_METAL;
            rec.mat.albedo = make_vec3(0.8f, 0.2f, 0.2f);
            rec.mat.fuzz = 0.0f;
            // rec.mat.shininess = 64.0f;
            rec.mat.shininess = 0.0f;
        }
    }
    return rec;
}

HYBRID_FUNC Vec3 shade(const Ray& r, const HitRecord& rec, const Light& light) {
    if (!rec.hit) {
        Vec3 unit_dir = unit_vector(r.dir);
        float t = 0.5f * (unit_dir.y + 1.0f);
        return make_vec3(1.0f, 1.0f, 1.0f)*(1.0f-t) + make_vec3(0.5f, 0.7f, 1.0f)*t;
    }

    // Ambient
    Vec3 ambient = rec.mat.albedo * 0.1f;

    // Diffuse
    Vec3 lightDir = unit_vector(light.position - rec.p);
    float diff = fmaxf(dot(rec.normal, lightDir), 0.0f);
    Vec3 diffuse = (rec.mat.albedo * light.color) * diff;

    // Specular
    Vec3 specular = make_vec3(0,0,0);
    if (rec.mat.type == MAT_METAL) {
        Vec3 viewDir = unit_vector(r.origin - rec.p);
        Vec3 halfDir = unit_vector(lightDir + viewDir);
        float spec = powf(fmaxf(dot(rec.normal, halfDir), 0.0f), rec.mat.shininess);
        specular = light.color * spec;
    }
    return ambient + diffuse + specular;
}

#endif