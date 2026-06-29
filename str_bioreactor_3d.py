#!/usr/bin/env python3
"""
Large-scale stirred-tank reactor (STR) — industrial 3D visualization.

Geometry follows common biopharma / cultured-meat STR conventions:
  - Cylindrical shell with torispherical bottom head (DIN-style)
  - Flat top head + drive housing
  - 4 vertical baffles (T/12 width, wall clearance)
  - 3 × 6-blade Rushton turbines on central shaft
  - Ring sparger, dip tube, sample / probe ports
  - 80 % liquid fill (optional scalar field from simulation NPZ)

Default size matches Bioreactor_OTR_model_compressible.py (~20 m³ class).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

# ---------------------------------------------------------------------------
# Geometry specification (typical large-scale STR, ~20 m³)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class STRGeometry:
    """Industrial STR proportions (H/D ≈ 1.6, Rushton impellers, 4 baffles)."""

    diameter: float = 2.516  # m — tank ID
    height: float = 4.026  # m — straight shell + heads (~1.6 × D)
    liquid_fill: float = 0.80
    n_impellers: int = 3
    n_blades: int = 6
    impeller_diameter_ratio: float = 0.38  # D_i / T  (common Rushton range 0.33–0.45)
    baffle_width_ratio: float = 1.0 / 12.0  # w_b / T
    baffle_clearance: float = 0.015  # m gap between baffle and wall
    shaft_diameter_ratio: float = 0.04  # d_s / T
    dish_depth_ratio: float = 0.15  # torispherical bottom depth / T
    top_head_height: float = 0.35  # m — flat head + motor skirt
    motor_diameter_ratio: float = 0.22
    motor_height: float = 0.55
    sparger_ring_diameter_ratio: float = 0.55
    sparger_elevation: float = 0.25  # m above dish apex

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def impeller_diameter(self) -> float:
        return self.impeller_diameter_ratio * self.diameter

    @property
    def shaft_radius(self) -> float:
        return 0.5 * self.shaft_diameter_ratio * self.diameter

    @property
    def baffle_width(self) -> float:
        return self.baffle_width_ratio * self.diameter

    @property
    def liquid_height(self) -> float:
        return self.liquid_fill * self.height

    @property
    def impeller_z_positions(self) -> list[float]:
        """Standard triple-Rushton spacing used in the CFD model."""
        return [0.30 * self.height, 0.50 * self.height, 0.65 * self.height]

    @property
    def volume_m3(self) -> float:
        return np.pi * self.radius**2 * self.height


# ---------------------------------------------------------------------------
# Materials (stainless steel + culture broth aesthetic)
# ---------------------------------------------------------------------------
MATERIALS = {
    "shell": {"color": "#BFC5CE", "metallic": 0.85, "roughness": 0.35, "opacity": 0.22},
    "shell_edge": {"color": "#8A9199", "metallic": 0.9, "roughness": 0.3, "opacity": 1.0},
    "dish": {"color": "#A8AEB6", "metallic": 0.88, "roughness": 0.32, "opacity": 0.28},
    "baffle": {"color": "#9DA4AD", "metallic": 0.82, "roughness": 0.38, "opacity": 0.85},
    "shaft": {"color": "#7E868F", "metallic": 0.92, "roughness": 0.25, "opacity": 1.0},
    "impeller": {"color": "#6E7680", "metallic": 0.95, "roughness": 0.22, "opacity": 1.0},
    "motor": {"color": "#4A5058", "metallic": 0.7, "roughness": 0.45, "opacity": 1.0},
    "sparger": {"color": "#5C8A8A", "metallic": 0.6, "roughness": 0.4, "opacity": 1.0},
    "liquid": {"color": "#4FC3F7", "metallic": 0.05, "roughness": 0.15, "opacity": 0.35},
    "port": {"color": "#707880", "metallic": 0.75, "roughness": 0.4, "opacity": 1.0},
    "field": {"opacity": 0.55},
}


def _apply_material(actor, spec: dict) -> None:
    prop = actor.GetProperty()
    prop.SetColor(pv.Color(spec["color"]).float_rgb)
    if "opacity" in spec:
        prop.SetOpacity(spec["opacity"])
    if hasattr(prop, "SetMetallic"):
        prop.SetMetallic(spec.get("metallic", 0.0))
    if hasattr(prop, "SetRoughness"):
        prop.SetRoughness(spec.get("roughness", 0.5))


def _merge(meshes: list[pv.PolyData | pv.UnstructuredGrid]) -> pv.PolyData:
    valid = [m for m in meshes if m is not None and m.n_points > 0]
    if not valid:
        return pv.PolyData()
    return valid[0].merge(valid[1:]) if len(valid) > 1 else valid[0]


def create_torispherical_bottom(geom: STRGeometry, resolution: int = 96) -> pv.PolyData:
    """Approximate 2:1 elliptical / torispherical bottom head."""
    r = geom.radius
    dish_h = geom.dish_depth_ratio * geom.diameter
    # Sphere segment forming the curved bottom
    sphere = pv.Sphere(radius=r * 1.05, center=(0.0, 0.0, r * 0.92), theta_resolution=resolution, phi_resolution=resolution)
    dish = sphere.clip("z", origin=(0.0, 0.0, dish_h * 0.35), invert=False)
    dish = dish.clip("z", origin=(0.0, 0.0, -0.05), invert=True)
    # Flat knuckle transition ring
    knuckle = pv.Cylinder(
        center=(0.0, 0.0, dish_h),
        direction=(0.0, 0.0, 1.0),
        radius=r,
        height=0.04,
        resolution=resolution,
    )
    return _merge([dish, knuckle])


def create_cylindrical_shell(geom: STRGeometry, resolution: int = 96) -> pv.PolyData:
    dish_h = geom.dish_depth_ratio * geom.diameter
    z0 = dish_h
    shell_h = geom.height - dish_h
    shell = pv.Cylinder(
        center=(0.0, 0.0, z0 + shell_h / 2.0),
        direction=(0.0, 0.0, 1.0),
        radius=geom.radius,
        height=shell_h,
        resolution=resolution,
    )
    return shell


def create_top_head(geom: STRGeometry, resolution: int = 64) -> tuple[pv.PolyData, pv.PolyData]:
    """Flat top head + agitator drive housing (typical industrial STR)."""
    z_top = geom.height
    head = pv.Cylinder(
        center=(0.0, 0.0, z_top + geom.top_head_height / 2.0),
        direction=(0.0, 0.0, 1.0),
        radius=geom.radius,
        height=geom.top_head_height,
        resolution=resolution,
    )
    motor_r = geom.motor_diameter_ratio * geom.diameter / 2.0
    motor = pv.Cylinder(
        center=(0.0, 0.0, z_top + geom.top_head_height + geom.motor_height / 2.0),
        direction=(0.0, 0.0, 1.0),
        radius=motor_r,
        height=geom.motor_height,
        resolution=resolution,
    )
    # Motor cap
    cap = pv.Cylinder(
        center=(0.0, 0.0, z_top + geom.top_head_height + geom.motor_height + 0.06),
        direction=(0.0, 0.0, 1.0),
        radius=motor_r * 1.08,
        height=0.12,
        resolution=resolution,
    )
    return head, _merge([motor, cap])


def create_baffles(geom: STRGeometry) -> pv.PolyData:
    dish_h = geom.dish_depth_ratio * geom.diameter
    z0 = dish_h + 0.05
    z1 = geom.liquid_height - 0.05
    baffle_h = max(z1 - z0, 0.5)
    baffle_r = geom.radius - geom.baffle_clearance - geom.baffle_width / 2.0
    baffles = []
    for angle in (0.0, 90.0, 180.0, 270.0):
        baffle = pv.Box(bounds=(
            baffle_r - geom.baffle_width / 2.0,
            baffle_r + geom.baffle_width / 2.0,
            -geom.baffle_width / 2.0,
            geom.baffle_width / 2.0,
            z0,
            z0 + baffle_h,
        ))
        baffle.rotate_z(angle, point=(0.0, 0.0, 0.0), inplace=True)
        baffles.append(baffle)
    return _merge(baffles)


def create_shaft(geom: STRGeometry, resolution: int = 48) -> pv.PolyData:
    z_bot = geom.dish_depth_ratio * geom.diameter * 0.5
    z_top = geom.height + geom.top_head_height * 0.5
    shaft = pv.Cylinder(
        center=(0.0, 0.0, (z_bot + z_top) / 2.0),
        direction=(0.0, 0.0, 1.0),
        radius=geom.shaft_radius,
        height=z_top - z_bot,
        resolution=resolution,
    )
    return shaft


def create_rushton_impeller(geom: STRGeometry, z: float, resolution: int = 64) -> pv.PolyData:
    """6-blade Rushton disk turbine — standard large-scale STR impeller."""
    r_disk = geom.impeller_diameter / 2.0
    r_inner = geom.shaft_radius * 1.15
    disk = pv.Disc(
        center=(0.0, 0.0, z),
        inner=r_inner,
        outer=r_disk,
        normal=(0.0, 0.0, 1.0),
        c_res=resolution,
    )
    blade_h = 0.14 * geom.impeller_diameter
    blade_w = 0.18 * geom.impeller_diameter
    blade_t = 0.025 * geom.impeller_diameter
    blades = []
    for i in range(geom.n_blades):
        angle = i * (360.0 / geom.n_blades)
        blade = pv.Box(bounds=(
            r_inner,
            r_disk,
            -blade_t / 2.0,
            blade_t / 2.0,
            z,
            z + blade_h,
        ))
        blade.rotate_z(angle, point=(0.0, 0.0, z), inplace=True)
        blades.append(blade)
    return _merge([disk, *blades])


def create_sparger_ring(geom: STRGeometry, resolution: int = 80) -> pv.PolyData:
    r_ring = geom.sparger_ring_diameter_ratio * geom.radius
    z = geom.sparger_elevation
    torus = pv.ParametricTorus(ringradius=r_ring, crosssectionradius=0.025, u_res=resolution, v_res=32)
    torus.translate((0.0, 0.0, z), inplace=True)
    # Dip tube from top
    dip = pv.Cylinder(
        center=(geom.radius * 0.35, 0.0, geom.liquid_height * 0.55),
        direction=(0.0, 0.0, 1.0),
        radius=0.035,
        height=geom.liquid_height * 0.5,
        resolution=24,
    )
    dip.rotate_z(12, point=(0.0, 0.0, 0.0), inplace=True)
    return _merge([torus, dip])


def create_probe_ports(geom: STRGeometry) -> pv.PolyData:
    """Sample / pH / DO probe nozzles on the shell (typical STR instrumentation)."""
    ports = []
    specs = [
        (geom.liquid_height * 0.55, 35.0, 0.18),
        (geom.liquid_height * 0.75, 145.0, 0.14),
        (geom.liquid_height * 0.35, 215.0, 0.12),
    ]
    for z, angle_deg, length in specs:
        nozzle = pv.Cylinder(
            center=(geom.radius + length / 2.0, 0.0, z),
            direction=(1.0, 0.0, 0.0),
            radius=0.045,
            height=length,
            resolution=24,
        )
        nozzle.rotate_z(angle_deg, point=(0.0, 0.0, 0.0), inplace=True)
        ports.append(nozzle)
    return _merge(ports)


def create_liquid_volume(geom: STRGeometry, resolution: int = 96) -> pv.PolyData:
    dish_h = geom.dish_depth_ratio * geom.diameter
    liquid = pv.Cylinder(
        center=(0.0, 0.0, (dish_h + geom.liquid_height) / 2.0),
        direction=(0.0, 0.0, 1.0),
        radius=geom.radius - 0.08,
        height=max(geom.liquid_height - dish_h, 0.1),
        resolution=resolution,
    )
    return liquid


def load_simulation_field(npz_path: Path, key: str = "C_O2") -> tuple[pv.StructuredGrid | None, str]:
    """Optional: colour liquid with a simulation field from final_fields.npz."""
    if not npz_path.is_file():
        return None, ""
    data = np.load(npz_path)
    if key not in data or "x" not in data or "mask" not in data:
        return None, ""
    x = data["x"]
    y = data["y"]
    z = data["z"]
    field = data[key]
    mask = data["mask"]
    nx, ny, nz = field.shape
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid = pv.StructuredGrid(xx, yy, zz)
    values = np.asarray(field, dtype=np.float64).copy()
    values[~mask] = np.nan
    grid[key] = values.flatten(order="F")
    labels = {
        "C_O2": "Dissolved O₂ [mol/m³]",
        "X_bio": "Biomass X [g/L]",
        "C_sub": "Substrate S [g/L]",
        "T": "Temperature [°C]",
        "OTR_local": "Local OTR [mol/(m³·s)]",
    }
    return grid, labels.get(key, key)


def build_str_reactor(
    geom: STRGeometry | None = None,
    field_npz: Path | None = None,
    field_key: str = "C_O2",
) -> tuple[pv.Plotter, dict]:
    geom = geom or STRGeometry()
    plotter = pv.Plotter(window_size=(1600, 1000), lighting="three lights")
    plotter.set_background("#E8EDF2", top="#F7F9FB")

    dish = create_torispherical_bottom(geom)
    shell = create_cylindrical_shell(geom)
    top_head, motor = create_top_head(geom)
    baffles = create_baffles(geom)
    shaft = create_shaft(geom)
    sparger = create_sparger_ring(geom)
    ports = create_probe_ports(geom)
    liquid = create_liquid_volume(geom)

    impellers = [create_rushton_impeller(geom, z) for z in geom.impeller_z_positions]

    # Tank envelope (semi-transparent stainless steel)
    for mesh, name in [
        (dish, "dish"),
        (shell, "shell"),
        (top_head, "shell"),
    ]:
        actor = plotter.add_mesh(mesh, smooth_shading=True, name=name)
        _apply_material(actor, MATERIALS["shell"])

    # Solid hardware
    for mesh, mat in [
        (baffles, "baffle"),
        (shaft, "shaft"),
        (motor, "motor"),
        (sparger, "sparger"),
        (ports, "port"),
    ]:
        actor = plotter.add_mesh(mesh, smooth_shading=True)
        _apply_material(actor, MATERIALS[mat])

    for imp in impellers:
        actor = plotter.add_mesh(imp, smooth_shading=True, show_edges=True, edge_color="#3A4048", line_width=0.6)
        _apply_material(actor, MATERIALS["impeller"])

    # Wireframe outline for shell readability
    outline = shell.extract_surface(algorithm="dataset_surface").extract_feature_edges(
        boundary_edges=True, feature_edges=False
    )
    plotter.add_mesh(outline, color="#6B7280", line_width=1.2, name="shell_edges")

    # Liquid + optional simulation scalar
    sim_grid, field_label = load_simulation_field(field_npz, field_key) if field_npz else (None, "")
    if sim_grid is not None and field_key in sim_grid.array_names:
        plotter.add_volume(
            sim_grid,
            scalars=field_key,
            opacity="sigmoid_6",
            cmap="plasma",
            show_scalar_bar=True,
        )
    else:
        actor = plotter.add_mesh(liquid, smooth_shading=True)
        _apply_material(actor, MATERIALS["liquid"])

    # Floor shadow disc
    floor = pv.Disc(inner=0, outer=geom.radius * 1.35, c_res=80, normal=(0, 0, 1))
    floor.translate((0, 0, -0.02), inplace=True)
    plotter.add_mesh(floor, color="#D1D5DB", opacity=0.35)

    # Annotations
    title = (
        f"Large-scale STR  |  V ≈ {geom.volume_m3:.1f} m³  |  "
        f"T = {geom.diameter:.2f} m  |  H = {geom.height:.2f} m  |  "
        f"{geom.n_impellers}× Rushton @ {geom.n_blades} blades"
    )
    plotter.add_text(title, position="upper_edge", font_size=11, color="#1F2937")
    plotter.add_text(
        f"Liquid fill {geom.liquid_fill:.0%}  |  "
        f"D_i/T = {geom.impeller_diameter_ratio:.2f}  |  "
        f"4 baffles  |  ring sparger",
        position=(0.02, 0.02),
        font_size=9,
        color="#374151",
        viewport=True,
    )
    if field_label:
        plotter.add_text(f"Field overlay: {field_label}", position=(0.02, 0.06), font_size=9, color="#374151", viewport=True)

    plotter.add_axes(line_width=2, labels_off=False)
    plotter.camera_position = [
        (geom.diameter * 2.8, -geom.diameter * 2.2, geom.height * 1.35),
        (0.0, 0.0, geom.height * 0.42),
        (0.0, 0.0, 1.0),
    ]

    meta = {
        "geometry": geom,
        "field_label": field_label,
        "volume_m3": geom.volume_m3,
    }
    return plotter, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial large-scale STR 3D model (PyVista)")
    parser.add_argument(
        "--field-npz",
        type=Path,
        default=Path("results/compressible_model/final_fields.npz"),
        help="Optional simulation NPZ for liquid scalar colouring",
    )
    parser.add_argument("--field-key", default="C_O2", help="Scalar array in NPZ (C_O2, X_bio, T, OTR_local, ...)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/str_bioreactor_3d.png"),
        help="Screenshot path",
    )
    parser.add_argument("--show", action="store_true", help="Open interactive window")
    parser.add_argument("--off-screen", action="store_true", help="Render without GUI")
    args = parser.parse_args()

    if args.off_screen:
        pv.OFF_SCREEN = True

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    npz = args.field_npz if args.field_npz.is_file() else None
    plotter, meta = build_str_reactor(field_npz=npz, field_key=args.field_key)

    if args.off_screen:
        plotter.render()
    else:
        plotter.show(auto_close=False)

    plotter.screenshot(str(args.output))
    print(f"STR 3D model saved: {args.output.resolve()}")
    print(f"  Volume class: {meta['volume_m3']:.1f} m3")
    if meta["field_label"]:
        print(f"  Scalar overlay: {meta['field_label'].replace(chr(8322), '2')}")

    if args.show and not args.off_screen:
        plotter.show()
    plotter.close()


if __name__ == "__main__":
    main()
