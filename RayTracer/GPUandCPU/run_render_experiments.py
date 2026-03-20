#!/usr/bin/env python3
import csv
import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
BUILD_DIR = BASE_DIR / "build"
BINARY_NAME = "bvh_viz"

# Experiment configuration (modifiable)
BASE_SCENE_JSON = BASE_DIR / "assets/json_files/sphere.json"

MAX_BOUNCES_LIST: List[int] = [2, 4, 8]
SPP_LIST: List[int] = [32, 64, 128]
RESOLUTIONS: List[Tuple[int, int]] = [
    (800, 600),
    (1280, 720),
    (1920, 1080),
]
REPEATS: int = 3

EXPERIMENTS_ROOT = BASE_DIR / "experiments"
RENDER_OUTPUT_NAME = "render.png"
CSV_FILENAME = "results.csv"
LOG_FILENAME = "render_logs.txt"

RENDER_TIMEOUT_SEC = 3600  # Per-render timeout in seconds, adjustable







def collect_system_info() -> Dict[str, Any]:
    """Collect CPU, memory, GPU information.

    On Slurm clusters, prefer the job's allocated memory (AllocTRES mem)
    over the node's physical memory from /proc/meminfo.
    """
    cpu_model = "N/A"
    mem_total_gb: Optional[float] = None
    gpu_name = "N/A"
    gpu_mem_total_mb: Optional[float] = None
    num_cpus: Optional[int] = None

    # CPU info
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        cpu_model = parts[1].strip()
                    break
    except Exception:
        pass

    # Memory info (fallback: node physical memory)
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    # Example: MemTotal:  131762548 kB
                    if len(parts) >= 2:
                        kb = float(parts[1])
                        mem_total_gb = kb / (1024 * 1024)
                    break
    except Exception:
        pass

    # If running under Slurm, prefer allocated memory (AllocTRES mem=...)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", slurm_job_id],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            if result.returncode == 0 and result.stdout:
                # Example line: AllocTRES=cpu=4,mem=24G,node=1,...
                m_mem = re.search(r"AllocTRES=[^\n]*?mem=([0-9]+)([MG]?)", result.stdout)
                if m_mem:
                    mem_value = float(m_mem.group(1))
                    unit = m_mem.group(2) or "G"
                    if unit.upper() == "G":
                        mem_total_gb = mem_value
                    elif unit.upper() == "M":
                        mem_total_gb = mem_value / 1024.0

                # Example line fragment: NumCPUs=4
                m_cpu = re.search(r"NumCPUs=([0-9]+)", result.stdout)
                if m_cpu:
                    try:
                        num_cpus = int(m_cpu.group(1))
                    except ValueError:
                        num_cpus = None
        except Exception:
            # If anything goes wrong, fall back to /proc/meminfo and os.cpu_count
            pass

    # Fallback: if Slurm info not available, use logical CPU count
    if num_cpus is None:
        try:
            detected = os.cpu_count()
            if detected is not None:
                num_cpus = int(detected)
        except Exception:
            num_cpus = None

    # GPU info (if multiple GPUs, only record the first one)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().splitlines()[0]
            # Example: NVIDIA A40, 45565 MiB
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 2:
                gpu_name = parts[0]
                mem_str = parts[1]
                m = re.search(r"([0-9.]+)", mem_str)
                if m:
                    gpu_mem_total_mb = float(m.group(1)) * 1024.0 / 1024.0  # MiB -> MB 近似
    except FileNotFoundError:
        # nvidia-smi not found
        pass
    except Exception:
        pass

    return {
        "cpu_model": cpu_model,
        "num_cpus": num_cpus if num_cpus is not None else "N/A",
        "memory_total_gb": mem_total_gb if mem_total_gb is not None else "N/A",
        "gpu_name": gpu_name,
        "gpu_memory_total_mb": gpu_mem_total_mb if gpu_mem_total_mb is not None else "N/A",
    }


def create_timestamped_output_dir() -> Path:
    ts = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    out_dir = EXPERIMENTS_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def load_scene_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_meshes_from_json(scene_data: Dict[str, Any]) -> int:
    scene = scene_data.get("scene", [])
    if not isinstance(scene, Iterable):
        return 0
    count = 0
    for obj in scene:
        if isinstance(obj, dict) and obj.get("type") == "mesh":
            count += 1
    return count


def write_temp_scene_json(
    base_scene_path: Path,
    base_scene_data: Dict[str, Any],
    max_bounces: int,
    spp: int,
    width: int,
    height: int,
) -> Path:
    """Override settings in memory and write a temporary JSON file."""
    data = deepcopy(base_scene_data)

    settings = data.setdefault("settings", {})
    settings["max_bounces"] = max_bounces
    settings["spp"] = spp

    camera = data.setdefault("camera", {})
    camera["pixel_width"] = width
    camera["pixel_height"] = height

    stem = base_scene_path.stem
    temp_name = f"{stem}_mb{max_bounces}_spp{spp}_{width}x{height}.json"
    temp_path = base_scene_path.parent / temp_name

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return temp_path


def cleanup_temp_json(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def parse_lbvh_and_render_times(stdout: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse LBVH Build Time and Render Time (ms) from stdout, compatible with CPU/GPU."""
    lbvh_time_ms: Optional[float] = None
    render_time_ms: Optional[float] = None

    m_lbvh = re.search(r"(CPU|GPU)\s+LBVH Build Time:\s*([0-9.]+)\s*ms", stdout)
    if m_lbvh:
        try:
            lbvh_time_ms = float(m_lbvh.group(2))
        except ValueError:
            lbvh_time_ms = None

    m_render = re.search(r"(CPU|GPU)\s+Render Time:\s*([0-9.]+)\s*ms", stdout)
    if m_render:
        try:
            render_time_ms = float(m_render.group(2))
        except ValueError:
            render_time_ms = None

    return lbvh_time_ms, render_time_ms


def parse_triangle_count(stdout: str) -> Optional[int]:
    """Parse triangle count from stdout (Scene stats line)."""
    m = re.search(r"Scene stats:\s*([0-9]+)\s+triangles", stdout)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def check_image_saved(stdout: str) -> bool:
    return "Image saved to render.png" in stdout


def run_renderer(scene_json_path: Path) -> Tuple[str, str, int]:
    """Invoke renderer in the build directory, returning stdout, stderr, returncode."""
    if not BUILD_DIR.is_dir():
        raise RuntimeError(f"BUILD_DIR does not exist: {BUILD_DIR}")

    # Use build directory as cwd, with json path relative
    rel_scene_path = os.path.relpath(scene_json_path, BUILD_DIR)

    # Use bash -lc to ensure the module command is available
    cmd = f"module load cuda-13.0.1-gcc-13.2.0; ./{BINARY_NAME} {rel_scene_path}"

    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=str(BUILD_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=RENDER_TIMEOUT_SEC,
    )
    return result.stdout, result.stderr, result.returncode


def move_render_image_to_output(
    output_dir: Path,
    base_scene_path: Path,
    max_bounces: int,
    spp: int,
    width: int,
    height: int,
    run_idx: int,
    global_idx: int,
) -> Optional[Path]:
    src = BUILD_DIR / RENDER_OUTPUT_NAME
    if not src.is_file():
        return None

    dst_name = (
        f"idx{global_idx:04d}_"
        f"{base_scene_path.stem}_mb{max_bounces}_spp{spp}"
        f"_{width}x{height}_run{run_idx}.png"
    )
    dst = output_dir / dst_name
    shutil.move(str(src), str(dst))
    return dst


def ensure_csv_header(csv_path: Path) -> None:
    if csv_path.is_file():
        return
    header = [
        "scene_json",
        "temp_json",
        "global_idx",
        "max_bounces",
        "spp",
        "resolution_width",
        "resolution_height",
        "triangle_count",
        "run_idx",
        "lbvh_build_time_ms",
        "render_time_ms",
        "cpu_model",
        "num_cpus",
        "memory_total_gb",
        "gpu_name",
        "gpu_memory_total_mb",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv_row(
    csv_path: Path,
    base_scene_path: Path,
    temp_scene_path: Path,
    global_idx: int,
    max_bounces: int,
    spp: int,
    width: int,
    height: int,
    triangle_count: Optional[int],
    run_idx: int,
    lbvh_time_ms: Optional[float],
    render_time_ms: Optional[float],
    sys_info: Dict[str, Any],
) -> None:
    ensure_csv_header(csv_path)
    row = [
        str(base_scene_path.relative_to(BASE_DIR)),
        temp_scene_path.name,
        global_idx,
        max_bounces,
        spp,
        width,
        height,
        triangle_count if triangle_count is not None else "",
        run_idx,
        lbvh_time_ms if lbvh_time_ms is not None else "",
        render_time_ms if render_time_ms is not None else "",
        sys_info.get("cpu_model", ""),
        sys_info.get("num_cpus", ""),
        sys_info.get("memory_total_gb", ""),
        sys_info.get("gpu_name", ""),
        sys_info.get("gpu_memory_total_mb", ""),
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main() -> None:
    print("Collecting system information...")
    sys_info = collect_system_info()
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    print("Creating experiment output directory...")
    output_dir = create_timestamped_output_dir()
    print(f"Output directory: {output_dir}")

    if not BASE_SCENE_JSON.is_file():
        raise FileNotFoundError(f"Base scene JSON does not exist: {BASE_SCENE_JSON}")

    print(f"Reading base scene: {BASE_SCENE_JSON}")
    base_scene_data = load_scene_json(BASE_SCENE_JSON)

    csv_path = output_dir / CSV_FILENAME
    log_path = output_dir / LOG_FILENAME

    total_configs = (
        len(MAX_BOUNCES_LIST) * len(SPP_LIST) * len(RESOLUTIONS) * REPEATS
    )
    current_idx = 0

    for max_bounces in MAX_BOUNCES_LIST:
        for spp in SPP_LIST:
            for width, height in RESOLUTIONS:
                temp_json_path = write_temp_scene_json(
                    BASE_SCENE_JSON,
                    base_scene_data,
                    max_bounces,
                    spp,
                    width,
                    height,
                )
                try:
                    for run_idx in range(1, REPEATS + 1):
                        current_idx += 1
                        print(
                            f"\n[{current_idx}/{total_configs}] "
                            f"mb={max_bounces}, spp={spp}, "
                            f"res={width}x{height}, run={run_idx}"
                        )
                        stdout, stderr, returncode = run_renderer(temp_json_path)

                        # Record command line output to log file to help debug parsing issues
                        with log_path.open("a", encoding="utf-8") as lf:
                            lf.write(
                                "=" * 40
                                + "\n"
                                + f"GLOBAL_IDX: {current_idx}\n"
                                + f"max_bounces={max_bounces}, spp={spp}, "
                                f"res={width}x{height}, run={run_idx}\n"
                                + f"returncode={returncode}\n"
                                + "--- STDOUT ---\n"
                            )
                            lf.write(stdout)
                            lf.write("\n--- STDERR ---\n")
                            lf.write(stderr)
                            lf.write("\n\n")

                        if returncode != 0:
                            print(
                                f"Render process returned non-zero exit code: {returncode}. "
                                f"stderr:\n{stderr}"
                            )

                        lbvh_time_ms, render_time_ms = parse_lbvh_and_render_times(stdout)
                        tri_count = parse_triangle_count(stdout)

                        if lbvh_time_ms is not None:
                            print(f"Parsed LBVH Build Time: {lbvh_time_ms} ms")
                        else:
                            print("Failed to parse LBVH Build Time line.")

                        if render_time_ms is not None:
                            print(f"Parsed Render Time: {render_time_ms} ms")
                        else:
                            print("Failed to parse Render Time line.")

                        if tri_count is not None:
                            print(f"Parsed triangle count: {tri_count}")
                        else:
                            print("Failed to parse triangle count line.")

                        if check_image_saved(stdout):
                            moved_path = move_render_image_to_output(
                                output_dir,
                                BASE_SCENE_JSON,
                                max_bounces,
                                spp,
                                width,
                                height,
                                run_idx,
                                current_idx,
                            )
                            if moved_path is not None:
                                print(f"Rendered image saved as: {moved_path}")
                            else:
                                print(
                                    "stdout indicates image was saved, but render.png was not found in the build directory."
                                )
                        else:
                            print("No image save message found in stdout.")

                        append_csv_row(
                            csv_path,
                            BASE_SCENE_JSON,
                            temp_json_path,
                            current_idx,
                            max_bounces,
                            spp,
                            width,
                            height,
                            tri_count,
                            run_idx,
                            lbvh_time_ms,
                            render_time_ms,
                            sys_info,
                        )
                finally:
                    cleanup_temp_json(temp_json_path)

    print("\nExperiment completed. Result CSV located at:")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()

