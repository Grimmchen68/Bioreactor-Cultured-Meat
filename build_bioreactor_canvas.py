#!/usr/bin/env python3
"""Build Cursor Canvas with embedded bioreactor simulation data."""

import json
from pathlib import Path

import numpy as np

NPZ = Path("results/compressible_model/final_fields.npz")
OUT = Path(
    r"C:\Users\katha\.cursor\projects\c-Users-katha-OneDrive-Desktop-PhD-Bioreactor-Cultured-Meat"
    r"\canvases\bioreactor-3d-explorer.canvas.tsx"
)

FIELD_META = [
    ("C_O2", "Dissolved O2", "mol/m3"),
    ("X_bio", "Biomass X", "g/L"),
    ("C_sub", "Substrate S", "g/L"),
    ("T", "Temperature", "degC"),
    ("OTR_local", "Local OTR", "mol/(m3 s)"),
    ("kLa", "kLa", "1/s"),
    ("O2_driving_force", "O2 driving force", "mol/m3"),
    ("OUR", "OUR", "mol/(m3 s)"),
    ("speed", "Flow speed", "m/s"),
    ("p", "Pressure", "Pa"),
    ("rho", "Density", "kg/m3"),
]


def main() -> None:
    data = np.load(NPZ)
    x, y, z = data["x"], data["y"], data["z"]
    mask = data["mask"]
    stride = 2

    voxels: list[list[float]] = []
    field_values: dict[str, list[float]] = {k: [] for k, _, _ in FIELD_META}

    for i in range(0, mask.shape[0], stride):
        for j in range(0, mask.shape[1], stride):
            for k in range(0, mask.shape[2], stride):
                if not mask[i, j, k]:
                    continue
                voxels.append([float(x[i]), float(y[j]), float(z[k])])
                for key, _, _ in FIELD_META:
                    field_values[key].append(float(data[key][i, j, k]))

    z_profile = []
    for k in range(mask.shape[2]):
        layer = mask[:, :, k]
        if not layer.any():
            continue
        entry = {"z": float(z[k])}
        for key, _, _ in FIELD_META:
            entry[key] = float(np.mean(data[key][:, :, k][layer]))
        z_profile.append(entry)

    payload = {
        "voxels": voxels,
        "fields": field_values,
        "zProfile": z_profile,
        "geometry": {
            "diameter": 2.516,
            "height": 4.026,
            "liquidFill": 0.8,
            "impellerZ": [0.30 * 4.026, 0.50 * 4.026, 0.65 * 4.026],
        },
        "fieldMeta": [{"key": k, "label": l, "unit": u} for k, l, u in FIELD_META],
    }

    voxels_json = json.dumps(voxels, separators=(",", ":"))
    fields_json = json.dumps(field_values, separators=(",", ":"))
    profile_json = json.dumps(z_profile, separators=(",", ":"))
    meta_json = json.dumps(payload["fieldMeta"], separators=(",", ":"))
    geom_json = json.dumps(payload["geometry"], separators=(",", ":"))

    canvas = CANVAS_TEMPLATE.replace("__VOXELS__", voxels_json)
    canvas = canvas.replace("__FIELDS__", fields_json)
    canvas = canvas.replace("__PROFILE__", profile_json)
    canvas = canvas.replace("__META__", meta_json)
    canvas = canvas.replace("__GEOM__", geom_json)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(canvas, encoding="utf-8")
    print(f"Wrote {OUT} ({len(voxels)} voxels)")


CANVAS_TEMPLATE = r'''import { useCallback, useMemo, useRef } from "react";
import {
  Button,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  LineChart,
  Row,
  Select,
  Stack,
  Stat,
  Text,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

const VOXELS: [number, number, number][] = __VOXELS__;
const FIELD_VALUES: Record<string, number[]> = __FIELDS__;
const Z_PROFILE: { z: number; [key: string]: number }[] = __PROFILE__;
const FIELD_META: { key: string; label: string; unit: string }[] = __META__;
const GEOM = __GEOM__ as {
  diameter: number;
  height: number;
  liquidFill: number;
  impellerZ: number[];
};

function plasma(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  const r = Math.min(255, Math.max(0, 255 * (1.5 * Math.abs(x - 0.75) - 0.5 + x)));
  const g = Math.min(255, Math.max(0, 255 * Math.sin(Math.PI * x)));
  const b = Math.min(255, Math.max(0, 255 * (1 - x) * (0.5 + x)));
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

function project(
  x: number,
  y: number,
  z: number,
  rotX: number,
  rotY: number,
  scale: number,
  cx: number,
  cy: number,
): [number, number, number] {
  const cosY = Math.cos(rotY);
  const sinY = Math.sin(rotY);
  const x1 = x * cosY - y * sinY;
  const y1 = x * sinY + y * cosY;
  const z1 = z;
  const cosX = Math.cos(rotX);
  const sinX = Math.sin(rotX);
  const y2 = y1 * cosX - z1 * sinX;
  const z2 = y1 * sinX + z1 * cosX;
  return [cx + x1 * scale, cy - z2 * scale, y2];
}

export default function Bioreactor3DExplorer() {
  const theme = useHostTheme();
  const [parameter, setParameter] = useCanvasState("parameter", "C_O2");
  const [rotX, setRotX] = useCanvasState("rotX", -0.45);
  const [rotY, setRotY] = useCanvasState("rotY", 0.65);
  const [zoom, setZoom] = useCanvasState("zoom", 95);
  const [viewMode, setViewMode] = useCanvasState<"voxel" | "slice">("viewMode", "voxel");
  const drag = useRef<{ x: number; y: number; active: boolean }>({ x: 0, y: 0, active: false });

  const meta = FIELD_META.find((f) => f.key === parameter) ?? FIELD_META[0];
  const values = FIELD_VALUES[parameter] ?? [];

  const stats = useMemo(() => {
    if (!values.length) return { min: 0, max: 0, mean: 0 };
    let min = values[0];
    let max = values[0];
    let sum = 0;
    for (const v of values) {
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    return { min, max, mean: sum / values.length };
  }, [values]);

  const points = useMemo(() => {
    const range = stats.max - stats.min || 1;
    const R = GEOM.diameter / 2;
    const cx = 360;
    const cy = 280;
    const projected = VOXELS.map(([x, y, z], i) => {
      const [sx, sy, depth] = project(x, y, z, rotX, rotY, zoom / R, cx, cy);
      const t = (values[i] - stats.min) / range;
      return { sx, sy, depth, color: plasma(t), z };
    });
    projected.sort((a, b) => a.depth - b.depth);
    return projected;
  }, [rotX, rotY, zoom, values, stats]);

  const sliceCells = useMemo(() => {
    const midY = 0;
    const tol = GEOM.diameter / 24;
    const range = stats.max - stats.min || 1;
    const cells: { x: number; z: number; t: number }[] = [];
    VOXELS.forEach(([x, , z], i) => {
      if (Math.abs(x) < 1e-6 || Math.abs(VOXELS[i][1] - midY) < tol) {
        cells.push({ x, z, t: (values[i] - stats.min) / range });
      }
    });
    return cells;
  }, [values, stats]);

  const onPointerDown = useCallback((e) => {
    drag.current = { x: e.clientX, y: e.clientY, active: true };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  }, []);

  const onPointerMove = useCallback(
    (e) => {
      if (!drag.current.active) return;
      const dx = e.clientX - drag.current.x;
      const dy = e.clientY - drag.current.y;
      drag.current = { x: e.clientX, y: e.clientY, active: true };
      setRotY((r) => r + dx * 0.008);
      setRotX((r) => Math.max(-1.2, Math.min(0.3, r + dy * 0.008)));
    },
    [setRotX, setRotY],
  );

  const onPointerUp = useCallback(() => {
    drag.current.active = false;
  }, []);

  const onWheel = useCallback(
    (e) => {
      setZoom((z) => Math.max(40, Math.min(160, z - e.deltaY * 0.05)));
    },
    [setZoom],
  );

  const profileSeries = useMemo(
    () => ({
      categories: Z_PROFILE.map((p) => p.z.toFixed(2)),
      data: Z_PROFILE.map((p) => Number(p[parameter]?.toFixed(4) ?? 0)),
    }),
    [parameter],
  );

  const tankR = GEOM.diameter / 2;
  const tankH = GEOM.height;
  const wire = useMemo(() => {
    const cx = 360;
    const cy = 280;
    const scale = zoom / tankR;
    const rings: { z: number; pts: string }[] = [0, tankH * GEOM.liquidFill, tankH].map((z) => {
      const seg: string[] = [];
      for (let a = 0; a <= 360; a += 12) {
        const rad = (a * Math.PI) / 180;
        const [sx, sy] = project(tankR * Math.cos(rad), tankR * Math.sin(rad), z, rotX, rotY, scale, cx, cy);
        seg.push(`${a === 0 ? "M" : "L"}${sx.toFixed(1)},${sy.toFixed(1)}`);
      }
      return { z, pts: seg.join(" ") + " Z" };
    });
    const impellers = GEOM.impellerZ.map((z) => {
      const [sx, sy] = project(0, 0, z, rotX, rotY, scale, cx, cy);
      const edge = project(tankR * 0.38, 0, z, rotX, rotY, scale, cx, cy);
      return { sx, sy, r: Math.abs(edge[0] - sx) };
    });
    return { rings, impellers, cx, cy };
  }, [rotX, rotY, zoom]);

  return (
    <Stack gap={16} style={{ padding: 16, maxWidth: 1200 }}>
      <Stack gap={4}>
        <H1>Large-scale STR bioreactor</H1>
        <Text style={{ color: theme.text.secondary }}>
          Interactive 3D view — drag to rotate, scroll to zoom. Select a parameter to colour the reactor volume.
        </Text>
      </Stack>

      <Grid columns="340px 1fr" gap={16}>
        <Card variant="outline">
          <CardHeader>Parameter</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Select
                value={parameter}
                onChange={setParameter}
                options={FIELD_META.map((f) => ({ value: f.key, label: `${f.label} [${f.unit}]` }))}
              />
              <Select
                value={viewMode}
                onChange={(v) => setViewMode(v as "voxel" | "slice")}
                options={[
                  { value: "voxel", label: "3D voxel cloud" },
                  { value: "slice", label: "Center X-Z slice" },
                ]}
              />
              <Row gap={8}>
                <Button onClick={() => { setRotX(-0.45); setRotY(0.65); setZoom(95); }}>Reset view</Button>
              </Row>
              <Text style={{ color: theme.text.tertiary, fontSize: 12 }}>
                V ≈ 20 m3 · T = {GEOM.diameter} m · H = {GEOM.height} m · 3 Rushton impellers
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Text style={{ color: theme.text.tertiary, fontSize: 12 }}>Source: final_fields.npz</Text>}>
            {`${meta.label} — ${viewMode === "voxel" ? "3D field" : "mid-plane slice"}`}
          </CardHeader>
          <CardBody>
            <svg
              width="100%"
              viewBox="0 0 720 520"
              style={{ background: theme.bg.editor, borderRadius: 8, border: `1px solid ${theme.stroke.secondary}`, touchAction: "none", cursor: "grab" }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerLeave={onPointerUp}
              onWheel={onWheel}
            >
              {wire.rings.map((ring, i) => (
                <path key={i} d={ring.pts} fill="none" stroke={theme.stroke.primary} strokeWidth={1.2} opacity={0.55} />
              ))}
              {wire.impellers.map((imp, i) => (
                <g key={i}>
                  <circle cx={imp.sx} cy={imp.sy} r={imp.r} fill="none" stroke={theme.text.secondary} strokeWidth={1.5} />
                  <line x1={wire.cx} y1={wire.cy - tankH * 0.5 * (zoom / tankR) * 0.01} x2={imp.sx} y2={imp.sy} stroke={theme.stroke.tertiary} strokeWidth={1} />
                </g>
              ))}
              {viewMode === "voxel"
                ? points.map((p, i) => (
                    <rect key={i} x={p.sx - 3.5} y={p.sy - 3.5} width={7} height={7} fill={p.color} opacity={0.85} />
                  ))
                : sliceCells.map((c, i) => {
                    const sx = 360 + c.x * (zoom / tankR);
                    const sy = 280 - c.z * (zoom / tankR);
                    return <rect key={i} x={sx - 4} y={sy - 4} width={8} height={8} fill={plasma(c.t)} opacity={0.9} />;
                  })}
              <text x={16} y={24} fill={theme.text.secondary} fontSize={11}>
                {meta.label} [{meta.unit}] · min {stats.min.toExponential(2)} · max {stats.max.toExponential(2)}
              </text>
              <rect x={600} y={16} width={16} height={120} fill={theme.fill.tertiary} stroke={theme.stroke.secondary} />
              {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                <rect key={t} x={600} y={16 + (1 - t) * 120} width={16} height={24} fill={plasma(t)} />
              ))}
              <text x={622} y={28} fill={theme.text.tertiary} fontSize={10}>high</text>
              <text x={622} y={140} fill={theme.text.tertiary} fontSize={10}>low</text>
            </svg>
          </CardBody>
        </Card>
      </Grid>

      <Grid columns={3} gap={12}>
        <Stat label="Minimum" value={stats.min.toExponential(3)} />
        <Stat label="Volume average" value={stats.mean.toExponential(3)} />
        <Stat label="Maximum" value={stats.max.toExponential(3)} />
      </Grid>

      <Card>
        <CardHeader>{`Axial profile — volume-averaged ${meta.label}`}</CardHeader>
        <CardBody>
          <Text style={{ color: theme.text.tertiary, fontSize: 11, marginBottom: 8 }}>
            Volume-averaged {meta.label} [{meta.unit}] vs axial position z [m]
          </Text>
          <LineChart
            categories={profileSeries.categories}
            series={[{ name: `${meta.label} [${meta.unit}]`, data: profileSeries.data, tone: "info" }]}
            height={220}
          />
        </CardBody>
      </Card>
    </Stack>
  );
}
'''

if __name__ == "__main__":
    main()
    
