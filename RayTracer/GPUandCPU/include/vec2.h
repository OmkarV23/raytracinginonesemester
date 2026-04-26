#ifndef VEC2_H
#define VEC2_H

#include <iostream>
#include <cmath>

#include "imports.h"

#ifdef __CUDACC__
    using Vec2 = float2; // Use CUDA's float2
#else
    struct float2_cpu { float x, y; };
    using Vec2 = float2_cpu; // Use custom CPU vector
#endif


HYBRID_FUNC inline Vec2 make_vec2(float x, float y) {
    Vec2 v; v.x = x; v.y = y; return v;
}
HYBRID_FUNC inline Vec2 vec2(float x, float y) {
    Vec2 v = make_vec2(x, y);
    return v;
};
HYBRID_FUNC inline Vec2 vec2(int x, int y) {
    Vec2 v = make_vec2(float(x), float(y));
    return v;
};
HYBRID_FUNC inline Vec2 point2(float x, float y) {
    Vec2 v = make_vec2(x, y);
    return v;
};
HYBRID_FUNC inline Vec2 point2(int x, int y) {
    Vec2 v = make_vec2(float(x), float(y));
    return v;
};

HYBRID_FUNC inline Vec2 operator+(const Vec2& a, const Vec2& b) { return make_vec2(a.x+b.x, a.y+b.y); }
HYBRID_FUNC inline Vec2 operator-(const Vec2& a, const Vec2& b) { return make_vec2(a.x-b.x, a.y-b.y); }
HYBRID_FUNC inline Vec2 operator-(const Vec2& a)                { return make_vec2(-a.x, -a.y); }
HYBRID_FUNC inline Vec2 operator*(const Vec2& a, const Vec2& b) { return make_vec2(a.x*b.x, a.y*b.y); }
HYBRID_FUNC inline Vec2 operator*(const Vec2& v, float t)       { return make_vec2(v.x*t, v.y*t); }
HYBRID_FUNC inline Vec2 operator*(float t, const Vec2& v)       { return make_vec2(v.x*t, v.y*t); }
HYBRID_FUNC inline Vec2 operator/(const Vec2& a, const Vec2& b) { return make_vec2(a.x/b.x, a.y/b.y); }
HYBRID_FUNC inline Vec2 operator/(const Vec2& a, double t)       { return make_vec2(a.x/t, a.y/t); }
HYBRID_FUNC inline float dot(const Vec2& u, const Vec2& v)      { return u.x*v.x + u.y*v.y; }
HYBRID_FUNC inline Vec2 cross(const Vec2& u, const Vec2& v) {
    return make_vec2(u.y * v.x - u.x * v.y,
                     u.x * v.y - u.y * v.x);
}
HYBRID_FUNC inline float length(const Vec2& v) { return sqrtf(dot(v, v)); }
HYBRID_FUNC inline float length_squared(const Vec2& v) { return dot(v, v); }
HYBRID_FUNC inline Vec2 normalize(const Vec2& v) { return v / length(v); }

HYBRID_FUNC inline Vec2 unit_vector(Vec2 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y);
    return make_vec2(v.x/len, v.y/len);
}

// Rec. 709 luminance — used for Russian Roulette survival probability
HYBRID_FUNC inline float luminance(const Vec2& c) {
    return 0.2126f * c.x + 0.7152f * c.y;
}

inline void PrintVec2(const Vec2& v) { std::cout << "(" << v.x << ", " << v.y << ")"; }

#endif