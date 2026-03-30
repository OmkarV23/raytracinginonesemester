// include/medium.h
#pragma once
#include "vec3.h"
#include "imports.h"

// ------------------------------------------------------------
// HomogeneousMedium
//
//   Free-path PDF:   p(t) = sigma_t * exp(-sigma_t * t)
//   Inverted CDF:    t    = -ln(1 - xi) / sigma_t         
//   Transmittance:   Tr   = exp(-sigma_t * t)              
//   Volume scatter throughput multiplier: sigma_s / sigma_t 
//   HG phase fn:     cos_theta = 1/(2g) * (1 + g^2 -
//                    ((1-g^2)/(1-g+2g*xi))^2)             
// ------------------------------------------------------------

struct HomogeneousMedium {
    float sigma_a = 0.0f;   // absorption coefficient  [m^-1]
    float sigma_s = 0.0f;   // scattering coefficient  [m^-1]
    float sigma_t = 0.0f;   // extinction = sigma_a + sigma_s
    float g       = 0.0f;   // Henyey-Greenstein anisotropy  (-1..1)
    bool  enabled = false;

    // Call after setting sigma_a / sigma_s to derive sigma_t
    HYBRID_FUNC inline void compute_sigma_t() {
        sigma_t = sigma_a + sigma_s;
    }

    // Single-scatter albedo: alpha = sigma_s / sigma_t  
    HYBRID_FUNC inline float albedo() const {
        return (sigma_t > 0.0f) ? (sigma_s / sigma_t) : 0.0f;
    }

    // Sample a free-path distance using inverted CDF:
    //   t = -ln(1 - xi) / sigma_t
    HYBRID_FUNC inline float sampleFreePath(float xi) const {
        if (sigma_t <= 0.0f) return 1e30f;
        if (xi >= 0.9999999f) xi = 0.9999999f;
        return -logf(1.0f - xi) / sigma_t;
    }

    // Transmittance Tr(t) = exp(-sigma_t * t)  
    HYBRID_FUNC inline float transmittance(float t) const {
        return expf(-sigma_t * t);
    }

    // Henyey-Greenstein phase function value 
    HYBRID_FUNC inline float phaseHG(float cos_theta) const {
        const float PI = 3.14159265358979f;
        if (fabsf(g) < 1e-4f) {
            return 1.0f / (4.0f * PI);
        }
        const float denom = 1.0f + g * g - 2.0f * g * cos_theta;
        return (1.0f - g * g) / (4.0f * PI * denom * sqrtf(denom));
    }

    // Sample a new direction from the HG distribution 
    HYBRID_FUNC inline Vec3 samplePhaseHG(const Vec3& wi_in,
                                           float xi1, float xi2) const {
        const float PI = 3.14159265358979f;

        float cos_theta;
        if (fabsf(g) < 1e-4f) {
            cos_theta = 1.0f - 2.0f * xi1;
        } else {
            const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi1);
            cos_theta = (1.0f + g * g - s * s) / (2.0f * g);
            cos_theta = fmaxf(-1.0f, fminf(1.0f, cos_theta));
        }

        const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
        const float phi       = 2.0f * PI * xi2;

        const Vec3 w = normalize(wi_in);
        Vec3 u, v;
        if (fabsf(w.x) > 0.9f)
            u = normalize(cross(make_vec3(0.0f, 1.0f, 0.0f), w));
        else
            u = normalize(cross(make_vec3(1.0f, 0.0f, 0.0f), w));
        v = cross(w, u);

        return u * (sin_theta * cosf(phi))
             + v * (sin_theta * sinf(phi))
             + w * cos_theta;
    }
};