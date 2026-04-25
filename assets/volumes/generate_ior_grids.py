#!/usr/bin/env python3
"""Generate IOR test grids for RRTE volumetric caustics scenes.

The Cornell box spans [-1,1]^3. The volume region should cover the
full box. The Luneburg sphere lens sits inside this volume at a
specified center and radius.
"""
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def write_f32_raw(filename, data):
    path = os.path.join(OUT_DIR, filename)
    with open(path, 'wb') as f:
        f.write(data.astype(np.float32).tobytes())
    print(f"  {path}: shape={data.shape}, min={data.min():.6f}, max={data.max():.6f}")

# Volume region bounds (covers the Cornell box plus epsilon)
VOL_MIN = np.array([-1.02, -1.02, -1.02])
VOL_MAX = np.array([ 1.02,  1.02,  1.02])
VOL_EXTENT = VOL_MAX - VOL_MIN  # 2.04 each

def world_to_grid(wx, wy, wz, nx, ny, nz):
    """Convert world coordinates to grid indices (float)."""
    u = (wx - VOL_MIN[0]) / VOL_EXTENT[0]
    v = (wy - VOL_MIN[1]) / VOL_EXTENT[1]
    w = (wz - VOL_MIN[2]) / VOL_EXTENT[2]
    return u * (nx-1), v * (ny-1), w * (nz-1)

def smoothstep(edge0, edge1, x):
    """Hermite smoothstep: 0 when x<=edge0, 1 when x>=edge1, smooth in between."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def sphere_smooth_weight(r_norm, transition_half=0.04):
    """Smooth inside/outside weight for a unit sphere (r_norm = r/R).
    Returns 1.0 deep inside, 0.0 well outside, smooth transition at surface.
    transition_half is in normalized radius units (~3 voxels at 128^3).
    """
    return 1.0 - smoothstep(1.0 - transition_half, 1.0 + transition_half, r_norm)

def generate_luneburg_ior(nx=128, ny=128, nz=128,
                          sphere_center=(0.0, 0.0, 0.0),
                          sphere_radius=0.4):
    """Luneburg lens: n(r) = sqrt(2 - (r/R)^2) inside sphere, n=1 outside.
    Grid stores n(r) - ior_base, so with ior_base=1.0, stores delta-n.
    At center: n=sqrt(2)~1.414, delta=0.414
    At surface: n=1.0, delta=0.0
    """
    print(f"\nLuneburg lens: center={sphere_center}, R={sphere_radius}")
    data = np.zeros((nz, ny, nx), dtype=np.float32)
    cx, cy, cz = sphere_center

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # World position of this voxel
                wx = VOL_MIN[0] + (ix / max(nx-1, 1)) * VOL_EXTENT[0]
                wy = VOL_MIN[1] + (iy / max(ny-1, 1)) * VOL_EXTENT[1]
                wz = VOL_MIN[2] + (iz / max(nz-1, 1)) * VOL_EXTENT[2]

                dx = (wx - cx) / sphere_radius
                dy = (wy - cy) / sphere_radius
                dz = (wz - cz) / sphere_radius
                r2 = dx*dx + dy*dy + dz*dz

                if r2 <= 1.0:
                    n = np.sqrt(max(2.0 - r2, 0.0))
                    data[iz, iy, ix] = n - 1.0  # delta from base
                # else: 0.0 (n = ior_base = 1.0)

    write_f32_raw(f"ior_luneburg_r{sphere_radius:.1f}_{nx}x{ny}x{nz}_f32.raw", data)
    return data

def generate_constant_sphere_ior(nx=128, ny=128, nz=128,
                                  sphere_center=(0.0, 0.0, 0.0),
                                  sphere_radius=0.4,
                                  n_inside=1.8):
    """Constant IOR sphere: n=n_inside inside, n=1 outside.
    Smooth boundary over ~3 voxels to avoid grid-aligned artifacts.
    Grid stores n-1 (delta from ior_base=1.0)."""
    print(f"\nConstant IOR sphere: n={n_inside}, center={sphere_center}, R={sphere_radius}")
    data = np.zeros((nz, ny, nx), dtype=np.float32)
    cx, cy, cz = sphere_center
    # Transition width in normalized radius: ~3 voxels
    voxel_r = VOL_EXTENT[0] / max(nx-1, 1) / sphere_radius
    transition = voxel_r * 3.0

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                wx = VOL_MIN[0] + (ix / max(nx-1, 1)) * VOL_EXTENT[0]
                wy = VOL_MIN[1] + (iy / max(ny-1, 1)) * VOL_EXTENT[1]
                wz = VOL_MIN[2] + (iz / max(nz-1, 1)) * VOL_EXTENT[2]

                dx = (wx - cx) / sphere_radius
                dy = (wy - cy) / sphere_radius
                dz = (wz - cz) / sphere_radius
                r_norm = np.sqrt(dx*dx + dy*dy + dz*dz)

                w = sphere_smooth_weight(r_norm, transition)
                if w > 1e-6:
                    data[iz, iy, ix] = w * (n_inside - 1.0)

    name = f"ior_const_n{n_inside:.1f}_r{sphere_radius:.1f}_{nx}x{ny}x{nz}_f32.raw"
    write_f32_raw(name, data)
    return data

def generate_scatter_mask(nx=128, ny=128, nz=128,
                          sphere_center=(0.0, 0.0, 0.0),
                          sphere_radius=0.4,
                          inside_value=0.0,
                          outside_value=1.0,
                          tag="glass"):
    """Generate a density/scatter mask grid.
    Controls where scattering happens in the volume:
      - Glass/lens:       inside=0, outside=1  (no scattering inside solid)
      - Translucent jelly: inside=1, outside=1  (scattering everywhere)
      - SSS object:       inside=5, outside=0  (dense scatter inside, no fog)
    Smooth boundary over ~3 voxels to avoid grid-aligned artifacts.
    Grid stores raw density values; sigma_s_effective = sigma_s * density.
    """
    print(f"\nScatter mask '{tag}': inside={inside_value}, outside={outside_value}, "
          f"center={sphere_center}, R={sphere_radius}")
    data = np.full((nz, ny, nx), outside_value, dtype=np.float32)
    cx, cy, cz = sphere_center
    voxel_r = VOL_EXTENT[0] / max(nx-1, 1) / sphere_radius
    transition = voxel_r * 3.0

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                wx = VOL_MIN[0] + (ix / max(nx-1, 1)) * VOL_EXTENT[0]
                wy = VOL_MIN[1] + (iy / max(ny-1, 1)) * VOL_EXTENT[1]
                wz = VOL_MIN[2] + (iz / max(nz-1, 1)) * VOL_EXTENT[2]

                dx = (wx - cx) / sphere_radius
                dy = (wy - cy) / sphere_radius
                dz = (wz - cz) / sphere_radius
                r_norm = np.sqrt(dx*dx + dy*dy + dz*dz)

                w = sphere_smooth_weight(r_norm, transition)
                # Blend: inside_value deep inside, outside_value far outside
                data[iz, iy, ix] = w * inside_value + (1.0 - w) * outside_value

    name = f"scatter_mask_{tag}_r{sphere_radius:.1f}_{nx}x{ny}x{nz}_f32.raw"
    write_f32_raw(name, data)
    return data


if __name__ == "__main__":
    R = 0.4   # sphere radius in world units
    C = (0.0, -0.15, 0.0)  # slightly below center so caustic beam hits floor
    N = 128   # grid resolution

    print("=== Generating RRTE volumetric caustics IOR grids ===")
    print(f"Volume bounds: {VOL_MIN} to {VOL_MAX}")
    print(f"Sphere center: {C}, radius: {R}")

    # Luneburg lens
    generate_luneburg_ior(N, N, N, sphere_center=C, sphere_radius=R)

    # Constant IOR spheres for comparison (like the paper)
    for n_val in [1.5, 1.8, 2.0, 2.1]:
        generate_constant_sphere_ior(N, N, N, sphere_center=C,
                                      sphere_radius=R, n_inside=n_val)

    # Scatter masks
    # Glass/lens: no scattering inside solid sphere, fog outside
    generate_scatter_mask(N, N, N, sphere_center=C, sphere_radius=R,
                          inside_value=0.0, outside_value=1.0, tag="glass")
    # Translucent: scattering everywhere (jelly, resin, etc.)
    generate_scatter_mask(N, N, N, sphere_center=C, sphere_radius=R,
                          inside_value=1.0, outside_value=1.0, tag="translucent")
    # SSS: dense scattering inside, no fog outside
    generate_scatter_mask(N, N, N, sphere_center=C, sphere_radius=R,
                          inside_value=5.0, outside_value=0.0, tag="sss")

    print("\nDone!")
