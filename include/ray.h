#ifndef RAY_H
#define RAY_H

#include "vec3.h"

struct Ray {
    using point3 = Vec3;
    
    point3 orig;
    Vec3 dir;

    HYBRID_FUNC Ray()
        : orig(make_vec3(0.0f, 0.0f, 0.0f)),
          dir(make_vec3(0.0f, 0.0f, 0.0f)) {}

    HYBRID_FUNC Ray(const point3& origin, const Vec3& direction)
        : orig(origin), dir(direction) {}

    HYBRID_FUNC inline point3 origin() const { return orig; }
    HYBRID_FUNC inline Vec3 direction() const { return dir; }
    HYBRID_FUNC inline point3 at(float t) const { return orig + dir * t; }
};

#endif
