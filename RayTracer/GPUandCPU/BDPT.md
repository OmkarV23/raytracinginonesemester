# Bidirectional Path Tracing (BDPT)

A Mitsuba/PBRT-style bidirectional path tracer for `RayTracer/GPUandCPU`. Replaces the unidirectional `TraceRayIterative` in the kernel; the rest of the pipeline (BVH, scene loading, denoiser, etc.) is unchanged.

## What's supported

**Surfaces**
- Diffuse + Blinn-Phong glossy (via the existing `EvaluateBRDF` / `SampleBRDF` / `BRDFpdf` helpers)
- Perfect mirror (`kr > 0`, `kd = ks = 0`, `ior = 1`)
- Dielectric / glass (`ior > 1`, `kd = ks = 0`) with Schlick-Fresnel + Snell + total-internal-reflection

**Participating media (`volume_region` in scene JSON)**
- Homogeneous regions: inverse-CDF free-flight sampling, closed-form transmittance
- Heterogeneous density grids: delta tracking against the density majorant, ratio-tracked transmittance for connection segments
- Henyey-Greenstein phase function with arbitrary `g`
- Optional emissive media: blackbody emission from temperature/flame channels accumulated along the camera walk

**BDPT machinery**
- Camera + light subpaths sharing a single `random_walk` that handles surfaces and medium scatter together
- `s = 0` (camera path hits emitter), `s = 1` (NEE-equivalent), `s ≥ 2` (full BDPT vertex connection)
- `t = 1` light tracing: each non-delta light vertex is projected onto the image plane and atomic-added to a splat buffer (helps caustics through delta surfaces and bright lights buried in dense media)
- Veach recursive-ratio MIS with power(2) heuristic, fwd/rev pdfs at every vertex; analytic distance pdf included for homogeneous medium vertices
- Per-strategy `BDPT_FIREFLY_CLAMP` (default `50`) bounds each contribution's luminance to suppress dense-media spikes; a sanitize pass zeros any `NaN`/`Inf` before accumulation

Out of scope: light tracing through point lights, non-pinhole cameras.

## Build

```bash
cd RayTracer/GPUandCPU
mkdir build && cd build
cmake -DENABLE_GPU=ON ..        # GPU build (requires CUDA)
make -j8
```

For a CPU-only build (slow but useful for debugging) drop `-DENABLE_GPU=ON`.

The executable is `bvh_viz` in the build dir.

## Run

```bash
./bvh_viz <scene.json> [-o output.png] [--nee-mode mis|brdf|area] [--denoise]
```

Scene file is the existing JSON format. `output.png` defaults to `render.png`. `--denoise` enables the OptiX AI denoiser (GPU only).

## Example scenes

The scenes under `assets/json_files/` cover the integrator's main regimes:

| Scene | What it exercises |
|---|---|
| `cornell_area_light.json` | Cornell box + area light + mirror sphere + emissive sphere — surface BDPT only |
| `cornell_volume_g0.json` | Glass spheres in homogeneous fog — dielectric + homogeneous medium |
| `cornell_smoke_taichi.json` | Cornell + heterogeneous smoke — delta tracking + ratio-tracked Tr |
| `cornell_smoke_embergen_120.json` | Dense embergen plume with hot emission — heterogeneous + volume emission |
| `cornell_smoke_embergen_47.json` | Plume with two emissive spheres at the base — t=1 light splatting earns its keep here |

```bash
# Surface-only Cornell box
./bvh_viz ../../assets/json_files/cornell_area_light.json -o cornell.png

# Homogeneous fog with glass
./bvh_viz ../../assets/json_files/cornell_volume_g0.json -o fog.png

# Dense smoke with hot emission
./bvh_viz ../../assets/json_files/cornell_smoke_embergen_120.json -o smoke.png
```

## Compile-time knobs

Define before `#include "bdpt.h"` to override defaults:

| Macro | Default | Effect |
|---|---|---|
| `BDPT_MAX_DEPTH` | `8` | Max number of vertices on each subpath. Path arrays are sized to `BDPT_MAX_DEPTH + 2`. The scene's `max_bounces` is clamped to this at the integrator entry. |
| `BDPT_FIREFLY_CLAMP` | `50.0f` | Per-`(s,t)` luminance clamp. `0` or negative disables. |

## Adding a volumetric scene

A `volume_region` entry in the scene JSON's `scene` array declares a participating medium:

```json
{
  "name": "fog",
  "type": "volume_region",
  "bounds_min": [-1, -1, -1],
  "bounds_max": [ 1,  1,  1],
  "medium": {
    "sigma_s": 0.6,
    "sigma_a": 0.01,
    "g": 0.0,

    "density_file":   "./assets/volumes/...raw",
    "density_resolution": [128, 128, 128],
    "density_format": "f32",
    "density_scale": 1.0,
    "majorant": 0.65,

    "emission_scale":      0.0,
    "emission_temperature": [400, 2200]
  }
}
```

`density_file` + the resolution/format triplet are optional — without them the region is homogeneous. `emission_scale > 0` activates blackbody emission from the temperature/flame channels (heterogeneous only).

## Notes on BDPT cost

BDPT is roughly 2–3× slower per sample than unidirectional PT (two random walks per sample, plus `s × t` connection shadow rays). Variance is lower per sample, particularly in dense media and around delta materials, so equal-time it tends to be competitive or better. The firefly clamp adds a small bounded bias in exchange for much lower variance — disable it via `-DBDPT_FIREFLY_CLAMP=0` if you need a pure unbiased reference.
