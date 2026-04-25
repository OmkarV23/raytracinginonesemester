// include/rrte.h
// Refractive Radiative Transfer Equation (RRTE) support
//
// Implements:
//   - Spatially-varying IOR field with gradient computation
//   - RK4 curved ray marching (eikonal ray equation)
//   - Delta tracking along curved paths
//   - Iterative NEE BVP solver (shooting method + Newton)
//   - Warp-cooperative BVP solver for GPU
//
// Reference: "Path Tracing Estimators for Refractive Radiative Transfer"
#pragma once

#include "vec3.h"
#include "ray.h"
#include "imports.h"
#include "medium.h"
#include "rng.h"

// ============================================================
// Configuration
// ============================================================
static constexpr int   RRTE_BVP_MAX_ITERS     = 8;      // Newton iterations for BVP
static constexpr int   RRTE_RK4_MAX_STEPS     = 2048;   // Max RK4 steps per ray segment
static constexpr float RRTE_BVP_TOL           = 1e-3f;  // BVP convergence tolerance (world units)
static constexpr float RRTE_GRAD_EPS          = 1e-3f;  // Central difference epsilon for grad(n)
static constexpr float RRTE_MIN_STEP          = 1e-5f;  // Minimum RK4 step size
static constexpr float RRTE_JACOBIAN_EPS      = 1e-4f;  // Finite-difference eps for Jacobian
static constexpr int   RRTE_WARP_SIZE         = 32;     // CUDA warp size
static constexpr int   RRTE_CONNECTION_MAX_RESTARTS = 12;
static constexpr float RRTE_CONNECTION_RR_WEIGHT    = 0.2f;
static constexpr float RRTE_CONNECTION_MATCH_COS    = 0.9995f;

// ============================================================
// RK4 ray state: position + optical momentum
// ============================================================
struct RayState {
    Vec3  x;    // position
    Vec3  d;    // optical momentum d = n(x) * omega (not normalized)
    float tau;  // accumulated optical depth
};

// ============================================================
// IOR field sampling (extends VolumeRegionGPU)
// These are free functions that take the IOR grid pointers directly.
// ============================================================

// Sample IOR at a point using trilinear interpolation on the IOR grid.
// If no IOR grid is provided, returns ior_base (uniform medium).
HYBRID_FUNC inline float sample_ior(
    const Vec3& p,
    const float* ior_data,
    int ior_nx, int ior_ny, int ior_nz,
    float ior_scale,
    float ior_base,
    const Vec3& min_bounds,
    const Vec3& max_bounds)
{
    if (ior_data == nullptr || ior_nx <= 0 || ior_ny <= 0 || ior_nz <= 0)
        return ior_base;

    const float extent_x = max_bounds.x - min_bounds.x;
    const float extent_y = max_bounds.y - min_bounds.y;
    const float extent_z = max_bounds.z - min_bounds.z;
    if (extent_x <= 1e-8f || extent_y <= 1e-8f || extent_z <= 1e-8f) return ior_base;

    const float u = fminf(fmaxf((p.x - min_bounds.x) / extent_x, 0.0f), 1.0f);
    const float v = fminf(fmaxf((p.y - min_bounds.y) / extent_y, 0.0f), 1.0f);
    const float w = fminf(fmaxf((p.z - min_bounds.z) / extent_z, 0.0f), 1.0f);

    const float gx = u * float(ior_nx - 1);
    const float gy = v * float(ior_ny - 1);
    const float gz = w * float(ior_nz - 1);

    const int x0 = int(floorf(gx));
    const int y0 = int(floorf(gy));
    const int z0 = int(floorf(gz));
    const int x1 = (x0 + 1 < ior_nx) ? x0 + 1 : x0;
    const int y1 = (y0 + 1 < ior_ny) ? y0 + 1 : y0;
    const int z1 = (z0 + 1 < ior_nz) ? z0 + 1 : z0;

    const float tx = gx - float(x0);
    const float ty = gy - float(y0);
    const float tz = gz - float(z0);

    auto voxel = [&](int ix, int iy, int iz) -> float {
        return ior_data[(iz * ior_ny + iy) * ior_nx + ix];
    };

    const float c000 = voxel(x0, y0, z0);
    const float c100 = voxel(x1, y0, z0);
    const float c010 = voxel(x0, y1, z0);
    const float c110 = voxel(x1, y1, z0);
    const float c001 = voxel(x0, y0, z1);
    const float c101 = voxel(x1, y0, z1);
    const float c011 = voxel(x0, y1, z1);
    const float c111 = voxel(x1, y1, z1);

    const float c00 = c000 * (1.0f - tx) + c100 * tx;
    const float c10 = c010 * (1.0f - tx) + c110 * tx;
    const float c01 = c001 * (1.0f - tx) + c101 * tx;
    const float c11 = c011 * (1.0f - tx) + c111 * tx;
    const float c0  = c00  * (1.0f - ty) + c10  * ty;
    const float c1  = c01  * (1.0f - ty) + c11  * ty;

    return ior_base + ior_scale * (c0 * (1.0f - tz) + c1 * tz);
}

// Compute gradient of IOR using central differences
HYBRID_FUNC inline Vec3 sample_grad_ior(
    const Vec3& p,
    const float* ior_data,
    int ior_nx, int ior_ny, int ior_nz,
    float ior_scale,
    float ior_base,
    const Vec3& min_bounds,
    const Vec3& max_bounds)
{
    // Use adaptive epsilon based on voxel size
    const float extent_x = max_bounds.x - min_bounds.x;
    const float extent_y = max_bounds.y - min_bounds.y;
    const float extent_z = max_bounds.z - min_bounds.z;

    // h = half a voxel in each dimension, clamped to RRTE_GRAD_EPS
    const float hx = fmaxf(extent_x / float(ior_nx > 1 ? ior_nx - 1 : 1) * 0.5f, RRTE_GRAD_EPS);
    const float hy = fmaxf(extent_y / float(ior_ny > 1 ? ior_ny - 1 : 1) * 0.5f, RRTE_GRAD_EPS);
    const float hz = fmaxf(extent_z / float(ior_nz > 1 ? ior_nz - 1 : 1) * 0.5f, RRTE_GRAD_EPS);

    const float nx = sample_ior(make_vec3(p.x + hx, p.y, p.z), ior_data, ior_nx, ior_ny, ior_nz, ior_scale, ior_base, min_bounds, max_bounds)
                   - sample_ior(make_vec3(p.x - hx, p.y, p.z), ior_data, ior_nx, ior_ny, ior_nz, ior_scale, ior_base, min_bounds, max_bounds);
    const float ny = sample_ior(make_vec3(p.x, p.y + hy, p.z), ior_data, ior_nx, ior_ny, ior_nz, ior_scale, ior_base, min_bounds, max_bounds)
                   - sample_ior(make_vec3(p.x, p.y - hy, p.z), ior_data, ior_nx, ior_ny, ior_nz, ior_scale, ior_base, min_bounds, max_bounds);
    const float nz = sample_ior(make_vec3(p.x, p.y, p.z + hz), ior_data, ior_nx, ior_ny, ior_nz, ior_scale, ior_base, min_bounds, max_bounds)
                   - sample_ior(make_vec3(p.x, p.y, p.z - hz), ior_data, ior_nx, ior_ny, ior_nz, ior_scale, ior_base, min_bounds, max_bounds);

    return make_vec3(nx / (2.0f * hx), ny / (2.0f * hy), nz / (2.0f * hz));
}

// ============================================================
// IOR field parameters (stored per VolumeRegionGPU)
// ============================================================
struct IORField {
    const float* ior_data = nullptr;
    int ior_nx = 0;
    int ior_ny = 0;
    int ior_nz = 0;
    float ior_scale = 1.0f;   // multiplier for grid values
    float ior_base  = 1.0f;   // base IOR (n = ior_base + ior_scale * grid_value)
    float ior_max_grad = 0.0f; // precomputed max |grad(n)| for step sizing

    HYBRID_FUNC inline bool has_ior_grid() const {
        return ior_data != nullptr && ior_nx > 0 && ior_ny > 0 && ior_nz > 0;
    }

    HYBRID_FUNC inline float sample(const Vec3& p, const Vec3& min_bounds, const Vec3& max_bounds) const {
        return sample_ior(p, ior_data, ior_nx, ior_ny, ior_nz, ior_scale, ior_base, min_bounds, max_bounds);
    }

    HYBRID_FUNC inline Vec3 gradient(const Vec3& p, const Vec3& min_bounds, const Vec3& max_bounds) const {
        return sample_grad_ior(p, ior_data, ior_nx, ior_ny, ior_nz, ior_scale, ior_base, min_bounds, max_bounds);
    }
};

// ============================================================
// RK4 Eikonal Ray Integrator
//
// The Hamiltonian ray equations for graded-index media:
//   dx/ds = d / n(x)           (position evolution)
//   dd/ds = grad(n(x))         (momentum evolution)
//
// where d = n(x) * omega is the optical momentum,
// omega is the unit direction, and s is the arc-length parameter.
// ============================================================

struct EikonalDerivatives {
    Vec3 dx;  // dx/ds
    Vec3 dd;  // dd/ds
};

// Compute dx/ds and dd/ds at a given state
HYBRID_FUNC inline EikonalDerivatives eikonal_rhs(
    const Vec3& x,
    const Vec3& d,
    const IORField& ior_field,
    const Vec3& min_bounds,
    const Vec3& max_bounds)
{
    const float n = ior_field.sample(x, min_bounds, max_bounds);
    const Vec3 grad_n = ior_field.gradient(x, min_bounds, max_bounds);

    EikonalDerivatives deriv;
    deriv.dx = d * (1.0f / fmaxf(n, 1e-6f));   // dx/ds = d/n
    deriv.dd = grad_n;                           // dd/ds = grad(n)
    return deriv;
}

// Single RK4 step
HYBRID_FUNC inline RayState rk4_step(
    const RayState& state,
    float ds,
    const IORField& ior_field,
    const HomogeneousMedium& medium,
    const Vec3& min_bounds,
    const Vec3& max_bounds,
    const float* density_data,
    int density_nx, int density_ny, int density_nz,
    float density_scale,
    float density_fallback)
{
    // k1
    const EikonalDerivatives k1 = eikonal_rhs(state.x, state.d, ior_field, min_bounds, max_bounds);

    // k2
    const Vec3 x2 = state.x + k1.dx * (0.5f * ds);
    const Vec3 d2 = state.d + k1.dd * (0.5f * ds);
    const EikonalDerivatives k2 = eikonal_rhs(x2, d2, ior_field, min_bounds, max_bounds);

    // k3
    const Vec3 x3 = state.x + k2.dx * (0.5f * ds);
    const Vec3 d3 = state.d + k2.dd * (0.5f * ds);
    const EikonalDerivatives k3 = eikonal_rhs(x3, d3, ior_field, min_bounds, max_bounds);

    // k4
    const Vec3 x4 = state.x + k3.dx * ds;
    const Vec3 d4 = state.d + k3.dd * ds;
    const EikonalDerivatives k4 = eikonal_rhs(x4, d4, ior_field, min_bounds, max_bounds);

    // Combine
    RayState next;
    next.x = state.x + (k1.dx + k2.dx * 2.0f + k3.dx * 2.0f + k4.dx) * (ds / 6.0f);
    next.d = state.d + (k1.dd + k2.dd * 2.0f + k3.dd * 2.0f + k4.dd) * (ds / 6.0f);

    // Accumulate optical depth along the step using midpoint density
    // Use the general trilinear sampling pattern matching VolumeRegionGPU
    const Vec3 mid = state.x + (k1.dx + k2.dx * 2.0f) * (ds / 4.0f);  // ~midpoint
    float density = density_fallback;
    if (density_data != nullptr && density_nx > 0 && density_ny > 0 && density_nz > 0) {
        // Inline trilinear for density at midpoint
        const float ext_x = max_bounds.x - min_bounds.x;
        const float ext_y = max_bounds.y - min_bounds.y;
        const float ext_z = max_bounds.z - min_bounds.z;
        if (ext_x > 1e-8f && ext_y > 1e-8f && ext_z > 1e-8f) {
            const float u = fminf(fmaxf((mid.x - min_bounds.x) / ext_x, 0.0f), 1.0f);
            const float v = fminf(fmaxf((mid.y - min_bounds.y) / ext_y, 0.0f), 1.0f);
            const float w = fminf(fmaxf((mid.z - min_bounds.z) / ext_z, 0.0f), 1.0f);
            const float gx = u * float(density_nx - 1);
            const float gy = v * float(density_ny - 1);
            const float gz = w * float(density_nz - 1);
            const int ix0 = int(floorf(gx)), iy0 = int(floorf(gy)), iz0 = int(floorf(gz));
            const int ix1 = (ix0 + 1 < density_nx) ? ix0 + 1 : ix0;
            const int iy1 = (iy0 + 1 < density_ny) ? iy0 + 1 : iy0;
            const int iz1 = (iz0 + 1 < density_nz) ? iz0 + 1 : iz0;
            const float tx = gx - float(ix0), ty = gy - float(iy0), tz = gz - float(iz0);

            auto vox = [&](int a, int b, int c) -> float {
                return density_data[(c * density_ny + b) * density_nx + a];
            };
            const float c00 = vox(ix0,iy0,iz0)*(1-tx) + vox(ix1,iy0,iz0)*tx;
            const float c10 = vox(ix0,iy1,iz0)*(1-tx) + vox(ix1,iy1,iz0)*tx;
            const float c01 = vox(ix0,iy0,iz1)*(1-tx) + vox(ix1,iy0,iz1)*tx;
            const float c11 = vox(ix0,iy1,iz1)*(1-tx) + vox(ix1,iy1,iz1)*tx;
            const float c0  = c00*(1-ty) + c10*ty;
            const float c1  = c01*(1-ty) + c11*ty;
            density = density_scale * (c0*(1-tz) + c1*tz);
        }
    }

    const float sigma_t_avg = spectrum_average(medium.sigma_t);
    next.tau = state.tau + sigma_t_avg * density * ds;

    return next;
}

// Compute adaptive step size based on IOR gradient magnitude
HYBRID_FUNC inline float compute_adaptive_step(
    const Vec3& x,
    const IORField& ior_field,
    const Vec3& min_bounds,
    const Vec3& max_bounds,
    float ds_base)
{
    const Vec3 grad_n = ior_field.gradient(x, min_bounds, max_bounds);
    const float grad_mag = sqrtf(dot(grad_n, grad_n));

    // Smaller steps where gradient is large
    if (grad_mag > 1e-6f) {
        const float ds_grad = 0.1f / grad_mag;  // curvature-limited step
        return fmaxf(RRTE_MIN_STEP, fminf(ds_base, ds_grad));
    }
    return ds_base;
}

// ============================================================
// Trace a curved ray through the IOR field using RK4
//
// Returns the final ray state after traversing the volume or
// reaching s_max arc length. Also accumulates optical depth (tau)
// for transmittance estimation.
// ============================================================
struct CurvedRayResult {
    RayState state;         // final position + momentum
    float    arc_length;    // total arc length traversed
    int      num_steps;     // number of RK4 steps taken
    bool     exited_volume; // did we leave the AABB?
    bool     scattered;     // did a delta-tracking scatter event occur?
    float    scatter_s;     // arc-length at scatter point (if scattered)
};

HYBRID_FUNC inline CurvedRayResult trace_curved_ray(
    const Vec3& x0,
    const Vec3& omega0,      // unit direction
    float s_max,              // max arc-length to traverse
    const IORField& ior_field,
    const HomogeneousMedium& medium,
    const Vec3& min_bounds,
    const Vec3& max_bounds,
    const float* density_data,
    int density_nx, int density_ny, int density_nz,
    float density_scale,
    float density_majorant,
    unsigned int& rng_state,
    bool do_delta_tracking,
    int tracking_channel)
{
    CurvedRayResult result;
    result.exited_volume = false;
    result.scattered = false;
    result.scatter_s = 0.0f;

    // Initial IOR at starting position
    const float n0 = ior_field.sample(x0, min_bounds, max_bounds);

    // Initialize ray state with optical momentum
    RayState state;
    state.x   = x0;
    state.d   = omega0 * n0;  // d = n * omega
    state.tau = 0.0f;

    // Compute base step size from voxel extent
    const float extent_min = fminf(max_bounds.x - min_bounds.x,
                             fminf(max_bounds.y - min_bounds.y,
                                   max_bounds.z - min_bounds.z));
    const int max_dim = (ior_field.ior_nx > 0) ?
        (ior_field.ior_nx > ior_field.ior_ny ?
            (ior_field.ior_nx > ior_field.ior_nz ? ior_field.ior_nx : ior_field.ior_nz) :
            (ior_field.ior_ny > ior_field.ior_nz ? ior_field.ior_ny : ior_field.ior_nz))
        : 64;
    const float ds_base = extent_min / float(max_dim) * 0.5f;

    // Delta tracking state
    const float inv_sigma_maj = (density_majorant > 1e-8f) ? (1.0f / density_majorant) : 0.0f;
    float next_free_path = 0.0f;
    if (do_delta_tracking && inv_sigma_maj > 0.0f) {
        const float xi = rng_next(rng_state);
        next_free_path = -logf(fmaxf(1.0f - xi, 1e-8f)) * inv_sigma_maj;
    }

    float s = 0.0f;
    int step = 0;
    for (; step < RRTE_RK4_MAX_STEPS && s < s_max; ++step) {
        const float ds = fminf(
            compute_adaptive_step(state.x, ior_field, min_bounds, max_bounds, ds_base),
            s_max - s);

        if (ds < RRTE_MIN_STEP * 0.1f) break;

        const RayState next = rk4_step(state, ds, ior_field, medium,
                                        min_bounds, max_bounds,
                                        density_data, density_nx, density_ny, density_nz,
                                        density_scale, 1.0f);

        s += ds;

        // Check if we've exited the volume AABB
        if (next.x.x < min_bounds.x || next.x.x > max_bounds.x ||
            next.x.y < min_bounds.y || next.x.y > max_bounds.y ||
            next.x.z < min_bounds.z || next.x.z > max_bounds.z)
        {
            result.exited_volume = true;
            // Use last valid position
            state = next;
            break;
        }

        state = next;

        // Delta tracking along curved path
        if (do_delta_tracking && inv_sigma_maj > 0.0f) {
            if (state.tau >= next_free_path) {
                // Potential scatter event - acceptance test
                float density = 1.0f;
                if (density_data != nullptr && density_nx > 0) {
                    // Sample density at current position using the volume grid
                    const float ext_x = max_bounds.x - min_bounds.x;
                    const float ext_y = max_bounds.y - min_bounds.y;
                    const float ext_z = max_bounds.z - min_bounds.z;
                    if (ext_x > 1e-8f && ext_y > 1e-8f && ext_z > 1e-8f) {
                        const float u = fminf(fmaxf((state.x.x - min_bounds.x) / ext_x, 0.0f), 1.0f);
                        const float v = fminf(fmaxf((state.x.y - min_bounds.y) / ext_y, 0.0f), 1.0f);
                        const float w = fminf(fmaxf((state.x.z - min_bounds.z) / ext_z, 0.0f), 1.0f);
                        const float gx = u * float(density_nx - 1);
                        const float gy = v * float(density_ny - 1);
                        const float gz = w * float(density_nz - 1);
                        const int ix0 = int(floorf(gx)), iy0 = int(floorf(gy)), iz0 = int(floorf(gz));
                        const int ix1 = (ix0+1 < density_nx) ? ix0+1 : ix0;
                        const int iy1 = (iy0+1 < density_ny) ? iy0+1 : iy0;
                        const int iz1 = (iz0+1 < density_nz) ? iz0+1 : iz0;
                        const float tx = gx-float(ix0), ty = gy-float(iy0), tz = gz-float(iz0);
                        auto vox = [&](int a, int b, int c) -> float {
                            return density_data[(c*density_ny+b)*density_nx+a];
                        };
                        const float c00=vox(ix0,iy0,iz0)*(1-tx)+vox(ix1,iy0,iz0)*tx;
                        const float c10=vox(ix0,iy1,iz0)*(1-tx)+vox(ix1,iy1,iz0)*tx;
                        const float c01=vox(ix0,iy0,iz1)*(1-tx)+vox(ix1,iy0,iz1)*tx;
                        const float c11=vox(ix0,iy1,iz1)*(1-tx)+vox(ix1,iy1,iz1)*tx;
                        const float c0=c00*(1-ty)+c10*ty;
                        const float c1=c01*(1-ty)+c11*ty;
                        density = density_scale * (c0*(1-tz)+c1*tz);
                    }
                }

                const Vec3 sigma_t_local = medium.sigma_t * density;
                const float accept = fminf(1.0f, fmaxf(0.0f,
                    spectrum_channel(sigma_t_local, tracking_channel) * inv_sigma_maj));
                if (rng_next(rng_state) < accept) {
                    result.scattered = true;
                    result.scatter_s = s;
                    break;
                }

                // Null collision: sample next free path
                const float xi = rng_next(rng_state);
                next_free_path = state.tau + (-logf(fmaxf(1.0f - xi, 1e-8f)) * inv_sigma_maj);
            }
        }
    }

    result.state = state;
    result.arc_length = s;
    result.num_steps = step;
    return result;
}

// ============================================================
// Get unit direction from optical momentum
// ============================================================
HYBRID_FUNC inline Vec3 momentum_to_direction(const Vec3& d, float n) {
    const float inv_n = 1.0f / fmaxf(n, 1e-6f);
    const Vec3 omega = d * inv_n;
    const float len = sqrtf(dot(omega, omega));
    return (len > 1e-8f) ? omega * (1.0f / len) : make_vec3(0.0f, 0.0f, 1.0f);
}

// ============================================================
// NEE BVP Solver: Shooting Method with Newton Iteration
//
// Given a scatter point x_s and a target light point x_L,
// find the initial direction omega_0 such that a curved ray
// from x_s arrives at x_L (within tolerance).
//
// Uses finite-difference Jacobian and Newton update on the
// angular parameterization (theta, phi) in a local frame.
// ============================================================

struct BVPResult {
    Vec3  omega_converged;     // converged initial direction (unit)
    float transmittance_tau;   // optical depth along converged path
    float endpoint_error;      // |x_end - x_L| at convergence
    float connection_weight;   // Mitsuba-style restart weight estimate
    bool  converged;           // did we converge?
    int   iterations_used;
};

// Convert direction to local (theta, phi) in a frame defined by (T, B, N_frame)
HYBRID_FUNC inline void direction_to_angles(
    const Vec3& omega,
    const Vec3& T, const Vec3& B, const Vec3& N_frame,
    float& theta, float& phi)
{
    const float cos_theta = fminf(fmaxf(dot(omega, N_frame), -1.0f), 1.0f);
    theta = acosf(cos_theta);
    const float proj_t = dot(omega, T);
    const float proj_b = dot(omega, B);
    phi = atan2f(proj_b, proj_t);
}

// Convert local (theta, phi) back to direction
HYBRID_FUNC inline Vec3 angles_to_direction(
    float theta, float phi,
    const Vec3& T, const Vec3& B, const Vec3& N_frame)
{
    const float sin_theta = sinf(theta);
    return T * (sin_theta * cosf(phi))
         + B * (sin_theta * sinf(phi))
         + N_frame * cosf(theta);
}

HYBRID_FUNC inline Vec3 rrte_uniform_sample_hemisphere(
    const Vec3& axis,
    unsigned int& rng_state)
{
    const float u1 = rng_next(rng_state);
    const float u2 = rng_next(rng_state);
    const float z = u1;
    const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    const float phi = 6.28318530717958647692f * u2;

    Vec3 T, B;
    if (fabsf(axis.x) > 0.9f) {
        T = normalize(cross(make_vec3(0.0f, 1.0f, 0.0f), axis));
    } else {
        T = normalize(cross(make_vec3(1.0f, 0.0f, 0.0f), axis));
    }
    B = cross(axis, T);

    Vec3 omega = T * (r * cosf(phi))
               + B * (r * sinf(phi))
               + axis * z;
    const float len = sqrtf(dot(omega, omega));
    return (len > 1e-8f) ? (omega * (1.0f / len)) : axis;
}

// Trace a curved ray and return endpoint
HYBRID_FUNC inline Vec3 trace_endpoint(
    const Vec3& x_start,
    const Vec3& omega,
    float s_max,
    const IORField& ior_field,
    const HomogeneousMedium& medium,
    const Vec3& min_bounds,
    const Vec3& max_bounds,
    const float* density_data,
    int density_nx, int density_ny, int density_nz,
    float density_scale,
    float& out_tau)
{
    unsigned int dummy_rng = 0;  // not doing delta tracking here
    CurvedRayResult res = trace_curved_ray(
        x_start, omega, s_max, ior_field, medium,
        min_bounds, max_bounds,
        density_data, density_nx, density_ny, density_nz,
        density_scale, 0.0f, dummy_rng, false, 0);
    out_tau = res.state.tau;
    return res.state.x;
}

// Scalar BVP solver (single-threaded Newton iteration)
HYBRID_FUNC inline BVPResult solve_nee_bvp_from_guess(
    const Vec3& x_scatter,
    const Vec3& x_light,
    const IORField& ior_field,
    const HomogeneousMedium& medium,
    const Vec3& min_bounds,
    const Vec3& max_bounds,
    const float* density_data,
    int density_nx, int density_ny, int density_nz,
    float density_scale,
    const Vec3& omega_init)
{
    BVPResult result;
    result.converged = false;
    result.iterations_used = 0;
    result.connection_weight = 1.0f;

    // Initial guess: straight-line direction
    Vec3 to_light = x_light - x_scatter;
    const float dist = sqrtf(dot(to_light, to_light));
    if (dist < 1e-6f) {
        result.omega_converged = make_vec3(0.0f, 1.0f, 0.0f);
        result.transmittance_tau = 0.0f;
        result.endpoint_error = 0.0f;
        result.converged = true;
        return result;
    }

    const Vec3 straight_dir = to_light * (1.0f / dist);
    Vec3 N_frame = omega_init;
    if (dot(N_frame, N_frame) < 1e-12f) {
        N_frame = straight_dir;
    } else {
        N_frame = normalize(N_frame);
    }

    // Build local frame
    Vec3 T, B;
    if (fabsf(N_frame.x) > 0.9f) {
        T = normalize(cross(make_vec3(0.0f, 1.0f, 0.0f), N_frame));
    } else {
        T = normalize(cross(make_vec3(1.0f, 0.0f, 0.0f), N_frame));
    }
    B = cross(N_frame, T);

    // Estimate arc-length budget generously enough for strongly bent paths.
    const float s_max = dist * 3.0f;

    // Start from the provided direction, expressed in the local frame.
    float theta = 0.0f;
    float phi   = 0.0f;
    direction_to_angles(N_frame, T, B, N_frame, theta, phi);
    float best_error = 1e30f;
    float best_theta = theta, best_phi = phi;
    float best_tau = 0.0f;

    for (int iter = 0; iter < RRTE_BVP_MAX_ITERS; ++iter) {
        result.iterations_used = iter + 1;

        const Vec3 omega = angles_to_direction(theta, phi, T, B, N_frame);

        float tau = 0.0f;
        const Vec3 x_end = trace_endpoint(x_scatter, omega, s_max, ior_field, medium,
                                           min_bounds, max_bounds,
                                           density_data, density_nx, density_ny, density_nz,
                                           density_scale, tau);

        const Vec3 delta = x_end - x_light;
        const float err = sqrtf(dot(delta, delta));

        if (err < best_error) {
            best_error = err;
            best_theta = theta;
            best_phi = phi;
            best_tau = tau;
        }

        if (err < RRTE_BVP_TOL) {
            result.converged = true;
            result.omega_converged = omega;
            result.transmittance_tau = tau;
            result.endpoint_error = err;
            return result;
        }

        // Finite-difference Jacobian: d(x_end) / d(theta, phi)
        // Perturb theta
        const float eps_t = RRTE_JACOBIAN_EPS;
        const float eps_p = RRTE_JACOBIAN_EPS;

        float tau_pt, tau_pp;
        const Vec3 omega_pt = angles_to_direction(theta + eps_t, phi, T, B, N_frame);
        const Vec3 x_pt = trace_endpoint(x_scatter, omega_pt, s_max, ior_field, medium,
                                          min_bounds, max_bounds,
                                          density_data, density_nx, density_ny, density_nz,
                                          density_scale, tau_pt);

        const Vec3 omega_pp = angles_to_direction(theta, phi + eps_p, T, B, N_frame);
        const Vec3 x_pp = trace_endpoint(x_scatter, omega_pp, s_max, ior_field, medium,
                                          min_bounds, max_bounds,
                                          density_data, density_nx, density_ny, density_nz,
                                          density_scale, tau_pp);

        // Jacobian columns (3x2 matrix)
        const Vec3 J_theta = (x_pt - x_end) * (1.0f / eps_t);
        const Vec3 J_phi   = (x_pp - x_end) * (1.0f / eps_p);

        // Solve 3x2 least-squares: J * d_angle = -delta
        // Using normal equations: (J^T J) d = -J^T delta
        const float a11 = dot(J_theta, J_theta);
        const float a12 = dot(J_theta, J_phi);
        const float a22 = dot(J_phi, J_phi);
        const float b1  = -dot(J_theta, delta);
        const float b2  = -dot(J_phi, delta);

        const float det = a11 * a22 - a12 * a12;
        if (fabsf(det) < 1e-12f) break;  // degenerate Jacobian

        const float inv_det = 1.0f / det;
        const float d_theta = (a22 * b1 - a12 * b2) * inv_det;
        const float d_phi   = (a11 * b2 - a12 * b1) * inv_det;

        // Damped Newton step (limit angular change to prevent divergence)
        const float max_step = 0.3f;
        const float step_norm = sqrtf(d_theta * d_theta + d_phi * d_phi);
        const float scale = (step_norm > max_step) ? (max_step / step_norm) : 1.0f;

        theta += d_theta * scale;
        phi   += d_phi * scale;

        // Clamp theta to [0, pi]
        theta = fmaxf(0.001f, fminf(3.14159f - 0.001f, theta));
    }

    // Return best found even if not converged
    result.omega_converged = angles_to_direction(best_theta, best_phi, T, B, N_frame);
    result.transmittance_tau = best_tau;
    result.endpoint_error = best_error;
    return result;
}

HYBRID_FUNC inline BVPResult solve_nee_bvp(
    const Vec3& x_scatter,
    const Vec3& x_light,
    const IORField& ior_field,
    const HomogeneousMedium& medium,
    const Vec3& min_bounds,
    const Vec3& max_bounds,
    const float* density_data,
    int density_nx, int density_ny, int density_nz,
    float density_scale)
{
    Vec3 to_light = x_light - x_scatter;
    const float dist2 = dot(to_light, to_light);
    const Vec3 omega_init = (dist2 > 1e-12f)
        ? (to_light * (1.0f / sqrtf(dist2)))
        : make_vec3(0.0f, 0.0f, 1.0f);
    return solve_nee_bvp_from_guess(
        x_scatter, x_light, ior_field, medium,
        min_bounds, max_bounds,
        density_data, density_nx, density_ny, density_nz,
        density_scale, omega_init);
}

HYBRID_FUNC inline BVPResult sample_rrte_connection(
    const Vec3& x_scatter,
    const Vec3& x_light,
    const IORField& ior_field,
    const HomogeneousMedium& medium,
    const Vec3& min_bounds,
    const Vec3& max_bounds,
    const float* density_data,
    int density_nx, int density_ny, int density_nz,
    float density_scale,
    unsigned int& rng_state)
{
    BVPResult result{};
    result.converged = false;
    result.connection_weight = 1.0f;
    result.endpoint_error = 1e30f;
    result.omega_converged = make_vec3(0.0f, 0.0f, 1.0f);
    result.transmittance_tau = 0.0f;
    result.iterations_used = 0;

    Vec3 to_light = x_light - x_scatter;
    const float dist2 = dot(to_light, to_light);
    if (dist2 < 1e-12f) {
        result.converged = true;
        result.endpoint_error = 0.0f;
        return result;
    }

    const Vec3 straight_dir = to_light * (1.0f / sqrtf(dist2));
    float weight = 1.0f;
    int successful_iterations = 1;
    Vec3 first_solution = make_vec3(0.0f, 0.0f, 1.0f);
    bool have_first_solution = false;

    for (int restart = 0; restart < RRTE_CONNECTION_MAX_RESTARTS; ++restart) {
        const Vec3 omega_init = (restart == 0)
            ? straight_dir
            : rrte_uniform_sample_hemisphere(straight_dir, rng_state);

        BVPResult candidate = solve_nee_bvp_from_guess(
            x_scatter, x_light, ior_field, medium,
            min_bounds, max_bounds,
            density_data, density_nx, density_ny, density_nz,
            density_scale, omega_init);

        if (candidate.endpoint_error < result.endpoint_error) {
            result = candidate;
        }

        if (candidate.converged || candidate.endpoint_error < RRTE_BVP_TOL * 5.0f) {
            if (!have_first_solution) {
                have_first_solution = true;
                first_solution = normalize(candidate.omega_converged);
                result = candidate;
                successful_iterations = 2;
                continue;
            }

            ++successful_iterations;
            const Vec3 omega_now = normalize(candidate.omega_converged);
            if (dot(first_solution, omega_now) > RRTE_CONNECTION_MATCH_COS) {
                result = candidate;
                result.converged = true;
                result.connection_weight = weight * float(successful_iterations - 1);
                return result;
            }
            continue;
        }

        if (rng_next(rng_state) < RRTE_CONNECTION_RR_WEIGHT) {
            weight *= 1.0f / RRTE_CONNECTION_RR_WEIGHT;
        } else {
            result.connection_weight = weight;
            return result;
        }
    }

    result.connection_weight = have_first_solution ? (weight * float(successful_iterations - 1)) : weight;
    return result;
}

// ============================================================
// Warp-Cooperative BVP Solver (GPU only)
//
// All threads in a warp collaborate on solving a single BVP.
// Each lane tries a different initial guess (spread over a cone),
// then a warp reduction picks the best guess for Newton refinement.
// ============================================================

#ifdef __CUDACC__

// Generate a perturbed initial guess for the given lane
__device__ inline Vec3 generate_warp_guess(
    int lane_id,
    const Vec3& straight_dir,
    float cone_half_angle)
{
    // Distribute guesses over a cone around straight_dir
    // Lane 0 = straight-line (no perturbation)
    if (lane_id == 0) return straight_dir;

    const float PI = 3.14159265358979f;
    // Map lane to (ring, position_in_ring)
    // Use a spiral pattern for good coverage
    const float t = float(lane_id) / float(RRTE_WARP_SIZE - 1);
    const float r = cone_half_angle * sqrtf(t);  // radius grows with sqrt for uniform area coverage
    const float golden_angle = PI * (3.0f - sqrtf(5.0f));  // ~2.39996
    const float phi = float(lane_id) * golden_angle;

    // Build perturbation in local frame of straight_dir
    Vec3 T, B;
    if (fabsf(straight_dir.x) > 0.9f)
        T = normalize(cross(make_vec3(0.0f, 1.0f, 0.0f), straight_dir));
    else
        T = normalize(cross(make_vec3(1.0f, 0.0f, 0.0f), straight_dir));
    B = cross(straight_dir, T);

    const float sin_r = sinf(r);
    Vec3 perturbed = straight_dir * cosf(r)
                   + T * (sin_r * cosf(phi))
                   + B * (sin_r * sinf(phi));

    const float len = sqrtf(dot(perturbed, perturbed));
    return (len > 1e-8f) ? perturbed * (1.0f / len) : straight_dir;
}

// Warp-cooperative BVP solve
// All 32 lanes participate. The requesting lane gets the result.
__device__ inline BVPResult warp_solve_nee_bvp(
    const Vec3& x_scatter,
    const Vec3& x_light,
    const IORField& ior_field,
    const HomogeneousMedium& medium,
    const Vec3& min_bounds,
    const Vec3& max_bounds,
    const float* density_data,
    int density_nx, int density_ny, int density_nz,
    float density_scale)
{
    const int lane = threadIdx.x % RRTE_WARP_SIZE;
    const unsigned int full_mask = 0xFFFFFFFFu;

    BVPResult result;
    result.converged = false;
    result.iterations_used = 0;

    Vec3 to_light = x_light - x_scatter;
    const float dist = sqrtf(dot(to_light, to_light));
    if (dist < 1e-6f) {
        result.omega_converged = make_vec3(0.0f, 1.0f, 0.0f);
        result.transmittance_tau = 0.0f;
        result.endpoint_error = 0.0f;
        result.converged = true;
        return result;
    }

    const Vec3 straight_dir = to_light * (1.0f / dist);
    const float s_max = dist * 1.5f;

    // Cone half-angle heuristic based on max IOR gradient
    float cone_angle = 0.1f;  // default ~6 degrees
    if (ior_field.ior_max_grad > 1e-6f) {
        cone_angle = fminf(0.5f, atanf(ior_field.ior_max_grad * dist));
    }

    // Phase 1: Each lane traces with its own initial guess
    const Vec3 my_guess = generate_warp_guess(lane, straight_dir, cone_angle);

    float my_tau = 0.0f;
    const Vec3 my_endpoint = trace_endpoint(
        x_scatter, my_guess, s_max, ior_field, medium,
        min_bounds, max_bounds,
        density_data, density_nx, density_ny, density_nz,
        density_scale, my_tau);

    const Vec3 my_delta = my_endpoint - x_light;
    float my_error = sqrtf(dot(my_delta, my_delta));

    // Phase 2: Warp reduction to find the lane with minimum error
    float min_error = my_error;
    int best_lane = lane;
    for (int offset = RRTE_WARP_SIZE / 2; offset > 0; offset /= 2) {
        const float other_error = __shfl_xor_sync(full_mask, min_error, offset);
        const int other_lane = __shfl_xor_sync(full_mask, best_lane, offset);
        if (other_error < min_error) {
            min_error = other_error;
            best_lane = other_lane;
        }
    }

    // Broadcast best lane to all threads
    best_lane = __shfl_sync(full_mask, best_lane, 0);
    min_error = __shfl_sync(full_mask, min_error, 0);

    // Get the best guess direction from the winning lane
    float best_gx = __shfl_sync(full_mask, my_guess.x, best_lane);
    float best_gy = __shfl_sync(full_mask, my_guess.y, best_lane);
    float best_gz = __shfl_sync(full_mask, my_guess.z, best_lane);
    float best_tau = __shfl_sync(full_mask, my_tau, best_lane);
    Vec3 best_guess = make_vec3(best_gx, best_gy, best_gz);

    // Phase 3: If already converged, we're done
    if (min_error < RRTE_BVP_TOL) {
        result.omega_converged = best_guess;
        result.transmittance_tau = best_tau;
        result.endpoint_error = min_error;
        result.converged = true;
        result.iterations_used = 0;
        return result;
    }

    // Phase 4: Newton refinement from the best guess (done by all lanes in lockstep)
    // Build local frame around best_guess for Newton iteration
    Vec3 N_frame = best_guess;
    Vec3 T, B;
    if (fabsf(N_frame.x) > 0.9f)
        T = normalize(cross(make_vec3(0.0f, 1.0f, 0.0f), N_frame));
    else
        T = normalize(cross(make_vec3(1.0f, 0.0f, 0.0f), N_frame));
    B = cross(N_frame, T);

    float theta = 0.0f, phi_angle = 0.0f;  // starting at the best guess = N_frame
    float current_best_error = min_error;
    float current_best_tau = best_tau;
    Vec3 current_best_omega = best_guess;

    // Use warp parallelism for Jacobian computation:
    // Lane 0: base point, Lane 1: theta+eps, Lane 2: phi+eps
    for (int iter = 0; iter < 4; ++iter) {  // fewer Newton iters since we have a good guess
        result.iterations_used = iter + 1;

        // All lanes compute the same base, theta-perturbed, and phi-perturbed traces
        // But we parallelize: lane%3 == 0 -> base, lane%3 == 1 -> theta, lane%3 == 2 -> phi
        const float eps = RRTE_JACOBIAN_EPS;
        float eval_theta = theta;
        float eval_phi = phi_angle;
        const int role = lane % 3;
        if (role == 1) eval_theta += eps;
        if (role == 2) eval_phi += eps;

        const Vec3 omega_eval = angles_to_direction(eval_theta, eval_phi, T, B, N_frame);
        float tau_eval = 0.0f;
        const Vec3 x_eval = trace_endpoint(x_scatter, omega_eval, s_max, ior_field, medium,
                                            min_bounds, max_bounds,
                                            density_data, density_nx, density_ny, density_nz,
                                            density_scale, tau_eval);
        const Vec3 delta_eval = x_eval - x_light;

        // Gather results from lanes 0, 1, 2 (using lane 0's group)
        const int group_base = (lane / 3) * 3;

        // Base endpoint (role 0)
        float base_dx = __shfl_sync(full_mask, delta_eval.x, group_base);
        float base_dy = __shfl_sync(full_mask, delta_eval.y, group_base);
        float base_dz = __shfl_sync(full_mask, delta_eval.z, group_base);
        float base_tau_val = __shfl_sync(full_mask, tau_eval, group_base);
        Vec3 delta_base = make_vec3(base_dx, base_dy, base_dz);
        float err_base = sqrtf(dot(delta_base, delta_base));

        // Theta-perturbed endpoint (role 1)
        float pt_dx = __shfl_sync(full_mask, delta_eval.x, group_base + 1);
        float pt_dy = __shfl_sync(full_mask, delta_eval.y, group_base + 1);
        float pt_dz = __shfl_sync(full_mask, delta_eval.z, group_base + 1);
        Vec3 delta_pt = make_vec3(pt_dx, pt_dy, pt_dz);

        // Phi-perturbed endpoint (role 2)
        float pp_dx = __shfl_sync(full_mask, delta_eval.x, group_base + 2);
        float pp_dy = __shfl_sync(full_mask, delta_eval.y, group_base + 2);
        float pp_dz = __shfl_sync(full_mask, delta_eval.z, group_base + 2);
        Vec3 delta_pp = make_vec3(pp_dx, pp_dy, pp_dz);

        if (err_base < current_best_error) {
            current_best_error = err_base;
            current_best_tau = base_tau_val;
            current_best_omega = angles_to_direction(theta, phi_angle, T, B, N_frame);
        }

        if (err_base < RRTE_BVP_TOL) {
            result.converged = true;
            break;
        }

        // Jacobian columns
        Vec3 J_theta = (delta_pt - delta_base) * (1.0f / eps);
        Vec3 J_phi   = (delta_pp - delta_base) * (1.0f / eps);

        // Normal equations
        const float a11 = dot(J_theta, J_theta);
        const float a12 = dot(J_theta, J_phi);
        const float a22 = dot(J_phi, J_phi);
        const float b1  = -dot(J_theta, delta_base);
        const float b2  = -dot(J_phi, delta_base);

        const float det = a11 * a22 - a12 * a12;
        if (fabsf(det) < 1e-12f) break;

        const float inv_det = 1.0f / det;
        float d_theta = (a22 * b1 - a12 * b2) * inv_det;
        float d_phi   = (a11 * b2 - a12 * b1) * inv_det;

        // Damped step
        const float max_step = 0.2f;
        const float step_norm = sqrtf(d_theta * d_theta + d_phi * d_phi);
        const float scale = (step_norm > max_step) ? (max_step / step_norm) : 1.0f;

        theta += d_theta * scale;
        phi_angle += d_phi * scale;
        theta = fmaxf(0.001f, fminf(3.14159f - 0.001f, theta));
    }

    result.omega_converged = current_best_omega;
    result.transmittance_tau = current_best_tau;
    result.endpoint_error = current_best_error;
    return result;
}

#endif  // __CUDACC__

// ============================================================
// Transmittance along a curved path
// ============================================================
HYBRID_FUNC inline Vec3 curved_transmittance(float tau, const HomogeneousMedium& medium) {
    // tau was accumulated using average sigma_t; reconstruct per-channel
    const float sigma_t_avg = spectrum_average(medium.sigma_t);
    if (sigma_t_avg < 1e-8f) return make_vec3(1.0f, 1.0f, 1.0f);

    return make_vec3(
        expf(-medium.sigma_t.x / sigma_t_avg * tau),
        expf(-medium.sigma_t.y / sigma_t_avg * tau),
        expf(-medium.sigma_t.z / sigma_t_avg * tau));
}
