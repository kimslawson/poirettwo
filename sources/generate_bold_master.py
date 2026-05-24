#!/usr/bin/env python3
"""
generate_bold_master.py
───────────────────────
Produces Poiret Two — a monolinear variable font derived from Poiret One.

Steps
  1. Monolinear correction on the Regular master
       Poiret One has very mild stroke contrast: vertical stems ≈ 33 units,
       horizontal strokes ≈ 29 units.  We correct the Regular to 33/33 by
       applying a small anisotropic expansion (delta_x=0, delta_y=2) so that
       horizontal strokes thicken by 2 units per side without touching stems.

  2. Bold master at wght=700
       Starting from the corrected Regular (33/33), apply a uniform miter-
       offset expansion (DELTA = 20 per side) in all directions.  Because
       the source is already monolinear, the Bold emerges monolinear too:
         vertical stems  : 33 + 2×20 = 73 units
         horizontal strokes: 33 + 2×20 = 73 units

  3. Variable-font metadata
       Axis: wght 400–700.  Instances: Regular/Medium/SemiBold/Bold.

Geometry
  • Anisotropic miter displacement: miter_displacement() is computed with
    delta=1 (unit vector), then the x- and y-components are scaled
    independently by delta_x / delta_y.
  • CCW (outer fill) paths → nodes move outward   → shape expands.
  • CW  (inner counter) paths → nodes move inward → counter contracts.
  • Nodes within SNAP units of a vertical metric (y = 0, 450, 750, −208,
    962) are snapped back so baseline/cap-height never drift.
  • For the monolinear correction a tighter snap radius is used so that
    interior stroke nodes (far from metrics) are corrected fully.
"""

import copy
import math
import re
import sys
import uuid
import glyphsLib
from glyphsLib.classes import (
    GSCustomParameter, GSFontMaster,
    GSLayer, GSNode, GSPath, GSAnchor, GSComponent,
    GSInstance,
)

# ── Configuration ──────────────────────────────────────────────────────
SOURCE       = "/home/user/poirettwo/sources/PoiretOne.glyphs"
OUTPUT       = "/home/user/poirettwo/sources/PoiretTwoVF.glyphs"
FONT_FAMILY  = "Poiret Two"

DELTA        = 20       # Bold expansion per side, font units
SNAP         = 3        # snap bold-y to metric when original is within SNAP
SNAP_STRICT  = 0.5      # snap radius used during the monolinear correction
                        # (only nodes essentially ON a metric are snapped)

# Monolinear correction for Regular
# delta_x = 0 : leave vertical stems untouched
# delta_y = 2 : thicken horizontal strokes by 2 units per side → 29 → 33
REG_DELTA_X  = 0
REG_DELTA_Y  = 2

# Vertical metrics to preserve across the weight axis
METRICS = [0, 450, 750, 962, -208]


# ── Geometry helpers ───────────────────────────────────────────────────

def right_normal(dx: float, dy: float) -> tuple[float, float]:
    """Unit normal pointing to the *right* of direction (dx, dy).
    right_normal(1, 0) → (0, -1);  right_normal(0, 1) → (1, 0)."""
    mag = math.hypot(dx, dy)
    if mag < 1e-9:
        return 0.0, 0.0
    return dy / mag, -dx / mag


def miter_displacement(
    positions: list[tuple[float, float]],
    i: int,
    delta: float,
    miter_limit: float = 6.0,
) -> tuple[float, float]:
    """
    Compute (dx, dy) for node at index i using the miter-offset rule.

    delta > 0 → move each node to the *right* of its direction of travel.
    This universally:
      • expands CCW (outer fill) contours outward
      • contracts CW (counter) contours inward
    Both effects make strokes bolder.

    miter_limit caps spikes at sharp concave corners.

    When delta=1.0 the return value is the unit miter-bisector vector,
    suitable for independent x/y scaling (anisotropic expansion).
    """
    n   = len(positions)
    p   = positions[i]
    pp  = positions[(i - 1) % n]
    pn  = positions[(i + 1) % n]

    dx1, dy1 = p[0] - pp[0], p[1] - pp[1]
    dx2, dy2 = pn[0] - p[0], pn[1] - p[1]
    m1 = math.hypot(dx1, dy1)
    m2 = math.hypot(dx2, dy2)

    if m1 < 1e-9 and m2 < 1e-9:
        return 0.0, 0.0
    if m1 < 1e-9:
        nx, ny = right_normal(dx2, dy2)
        return delta * nx, delta * ny
    if m2 < 1e-9:
        nx, ny = right_normal(dx1, dy1)
        return delta * nx, delta * ny

    n1x, n1y = right_normal(dx1, dy1)
    n2x, n2y = right_normal(dx2, dy2)

    bx, by = n1x + n2x, n1y + n2y
    bm = math.hypot(bx, by)
    if bm < 1e-9:
        return delta * n1x, delta * n1y
    bx /= bm
    by /= bm

    proj = bx * n1x + by * n1y
    if abs(proj) < 1e-9:
        return delta * n1x, delta * n1y

    scale = delta / proj
    if abs(scale) > miter_limit * abs(delta):
        scale = math.copysign(miter_limit * abs(delta), scale)

    return scale * bx, scale * by


def snap_y(orig_y: float, new_y: float, radius: float) -> float:
    """Snap new_y to the nearest alignment metric if orig_y is within radius."""
    for m in METRICS:
        if abs(orig_y - m) <= radius:
            return float(m)
    return new_y


def _compute_y_scales(
    positions: list[tuple[float, float]],
    snap_radius: float,
) -> list[float]:
    """
    Return a per-node y-scale factor (1.0 or 2.0).

    Strokes whose outer edge sits exactly on an alignment metric (e.g. the
    top bar of '7' touching y=750, or the base of '2' touching y=0) can only
    expand on one side — the metric side is snapped and stays put.  To reach
    the correct stroke thickness the free (inner) edge must travel 2× the
    nominal delta.

    Detection rule
    ─────────────
    For each node N not itself on a metric:
      • Examine both adjacent nodes P in the path.
      • If P is within snap_radius of a metric AND edge N→P is predominantly
        vertical (|Δx| / |Δy| < 0.5), mark y_scale[N] = 2.0.

    Propagation
    ───────────
    After the initial pass, nodes that share the same original y (within 3
    units) as a marked node inherit y_scale=2.0.  This keeps horizontal
    stroke faces flat (both endpoints of a crossbar bottom move equally).
    """
    n = len(positions)
    y_scale = [1.0] * n

    # ── Pass 1: direct detection ───────────────────────────────────────
    # MAX_STROKE_Y caps the vertical-edge length we consider "a stroke".
    # Horizontal strokes in Poiret Two are ~29–73 units tall.
    # Long vertical edges (stems, descenders) are 200+ units — we must
    # NOT flag those, or crossbar corners at y=0 would get doubled delta.
    MAX_STROKE_Y = 100

    for i in range(n):
        ox, oy = positions[i]
        # Skip if this node itself is on a metric — it will be snapped
        if any(abs(oy - m) <= snap_radius for m in METRICS):
            continue
        for di in (-1, +1):
            j   = (i + di) % n
            pox, poy = positions[j]
            # Neighbor must be metric-snapped (within snap_radius of a metric)
            if not any(abs(poy - m) <= snap_radius for m in METRICS):
                continue
            dx = abs(pox - ox)
            dy = abs(poy - oy)
            if dy < 5 or dy > MAX_STROKE_Y:
                continue          # degenerate or too long (a stem, not a stroke)
            if dx > dy * 0.5:
                continue          # too diagonal — not a clean vertical edge
            y_scale[i] = 2.0
            break

    # ── Pass 2: propagate along same-y groups ─────────────────────────
    # Any node within 3 units of a marked node's y inherits y_scale=2.0,
    # so the inner face of a horizontal stroke moves uniformly.
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if y_scale[i] != 2.0:
                continue
            oy_i = positions[i][1]
            for j in range(n):
                if y_scale[j] == 2.0:
                    continue
                oy_j = positions[j][1]
                if abs(oy_j - oy_i) < 3:
                    # Same-y group — if j is also on the "inside" (not at a metric)
                    if not any(abs(oy_j - m) <= snap_radius for m in METRICS):
                        y_scale[j] = 2.0
                        changed = True

    return y_scale


# ── Path expansion ─────────────────────────────────────────────────────

def expand_path(
    orig_path: GSPath,
    delta_x: float,
    delta_y: float,
    snap_radius: float = SNAP,
) -> GSPath:
    """
    Anisotropic miter-offset expansion with metric-boundary compensation.

    delta_x  expansion applied to the x-component of each node's miter
             displacement.  Controls horizontal movement = vertical-stem
             thickness.
    delta_y  expansion applied to the y-component.  Controls vertical
             movement = horizontal-stroke thickness.

    Metric-boundary compensation
    ─────────────────────────────
    When a horizontal stroke has one face on a y-metric (baseline, cap
    height …), that face is snapped and cannot move.  The opposite (inner)
    face must travel 2×delta_y to reach the target stroke thickness.
    _compute_y_scales() detects these nodes and returns y_scale=2.0 for
    them; the doubled delta is applied here.

    For the monolinear Regular correction: delta_x=0, delta_y=2
    For the Bold expansion: delta_x=delta_y=DELTA (isotropic)
    """
    positions = [(nd.position.x, nd.position.y) for nd in orig_path.nodes]
    types     = [nd.type   for nd in orig_path.nodes]
    smooths   = [nd.smooth for nd in orig_path.nodes]

    new_path        = GSPath()
    new_path.closed = orig_path.closed
    new_positions   = []
    new_nodes       = []

    # Per-node y-scale factors (1.0 normal, 2.0 for metric-bounded inner faces)
    y_scales = _compute_y_scales(positions, snap_radius)

    for i, (ox, oy) in enumerate(positions):
        if delta_x == 0.0 and delta_y == 0.0:
            nx, ny = ox, oy
        else:
            ux, uy = miter_displacement(positions, i, 1.0)
            ddx    = ux * delta_x
            ddy    = uy * delta_y * y_scales[i]
            nx     = ox + ddx
            ny     = snap_y(oy, oy + ddy, snap_radius)
        new_positions.append((nx, ny))

    # Post-process: fix co-linear nodes on metrics that got stuck during Bold
    if abs(delta_x) > 0.5 or abs(delta_y) > 0.5:
        new_positions = _fix_colinear_metric_nodes(
            positions, new_positions, delta_x
        )

    for i, (nx, ny) in enumerate(new_positions):
        node = GSNode((nx, ny), type=types[i], smooth=smooths[i])
        new_nodes.append(node)

    new_path.nodes = new_nodes
    return new_path


def _fix_colinear_metric_nodes(
    orig: list[tuple[float, float]],
    new:  list[tuple[float, float]],
    delta: float,
) -> list[tuple[float, float]]:
    """
    Correct nodes that sit on a y-metric with co-linear neighbours at the
    same y, causing the miter to produce a vertical displacement that snapping
    then cancels — leaving the node laterally unmoved.

    Example: the bottom-right inner corner of 'B' at (111, 0) has neighbours
    both at y=0, so the stem base fails to widen.  We nudge it laterally
    toward the path centroid by |delta|.
    """
    n          = len(orig)
    centroid_x = sum(p[0] for p in orig) / n
    fixed      = list(new)

    for i in range(n):
        ox, oy = orig[i]
        nx, ny = new[i]

        # Only truly stuck nodes
        if abs(nx - ox) > 0.5 or abs(ny - oy) > 0.5:
            continue
        # Only nodes on a metric
        if not any(abs(oy - m) <= SNAP for m in METRICS):
            continue
        # Both neighbours also on the same metric y
        p_prev = orig[(i - 1) % n]
        p_next = orig[(i + 1) % n]
        if abs(p_prev[1] - oy) > SNAP or abs(p_next[1] - oy) > SNAP:
            continue
        # Nudge toward centroid
        if ox < centroid_x - 1:
            fixed[i] = (ox + abs(delta), oy)
        elif ox > centroid_x + 1:
            fixed[i] = (ox - abs(delta), oy)

    return fixed


# ── Layer helpers ──────────────────────────────────────────────────────

def apply_correction_to_layer(layer: GSLayer) -> None:
    """
    In-place monolinear correction: expand horizontal strokes by
    REG_DELTA_Y per side without touching vertical stems (REG_DELTA_X=0).
    """
    orig_paths = list(layer.paths)
    # Remove existing paths
    while len(layer.paths):
        layer.paths.pop(0)
    # Re-add corrected paths
    for p in orig_paths:
        layer.paths.append(
            expand_path(p, REG_DELTA_X, REG_DELTA_Y, snap_radius=SNAP_STRICT)
        )


def make_bold_layer(reg_layer: GSLayer, bold_master_id: str) -> GSLayer:
    """
    Build the Bold GSLayer from a (monolinear-corrected) Regular layer.
    Uniform DELTA expansion produces monolinear Bold strokes.
    """
    bl                    = GSLayer()
    bl.layerId            = bold_master_id
    bl.associatedMasterId = bold_master_id
    bl.name               = "Bold"
    bl.width              = reg_layer.width  # advance width: SBs narrow naturally

    for path in reg_layer.paths:
        bl.paths.append(expand_path(path, DELTA, DELTA, snap_radius=SNAP))

    for comp in reg_layer.components:
        bl.components.append(comp.clone())

    for anch in reg_layer.anchors:
        a            = GSAnchor()
        a.name       = anch.name
        a.position.x = anch.position.x
        a.position.y = anch.position.y
        bl.anchors.append(a)

    return bl


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print(f"Loading  {SOURCE}")
    font = glyphsLib.load(SOURCE)

    reg_master = font.masters[0]
    reg_id     = reg_master.id
    print(f"Regular master: '{reg_master.name}'  id={reg_id}  wght={reg_master.weightValue}")
    print(f"Glyphs: {len(font.glyphs)}")

    # ── 0. Rename font family ──────────────────────────────────────────
    font.familyName = FONT_FAMILY
    print(f"Font family renamed to: {FONT_FAMILY!r}")

    # ── 0b. Monolinear correction on the Regular master ────────────────
    # Equalize stroke contrast: horizontal strokes (29 units) → 33 units
    # by expanding them by REG_DELTA_Y=2 per side.  Vertical stems are
    # left at 33 units (REG_DELTA_X=0).
    reg_corrected = 0
    for glyph in font.glyphs:
        layer = glyph.layers[reg_id]
        if layer is None or not layer.paths:
            continue
        apply_correction_to_layer(layer)
        reg_corrected += 1
    print(f"Monolinear correction applied to {reg_corrected} Regular layers")

    # ── 1. Stamp explicit axis location on the Regular master ─────────
    reg_master.customParameters.append(
        GSCustomParameter("Axis Location", [{"Axis": "Weight", "Location": 400}])
    )

    # ── 2. Create Bold master ──────────────────────────────────────────
    bold_master              = GSFontMaster()
    bold_master.name         = "Bold"
    bold_master.weightValue  = 700
    bold_master.widthValue   = reg_master.widthValue

    bold_master.ascender     = reg_master.ascender
    bold_master.descender    = reg_master.descender
    bold_master.capHeight    = reg_master.capHeight
    bold_master.xHeight      = reg_master.xHeight
    bold_master.alignmentZones = list(reg_master.alignmentZones)

    # Stem values updated for monolinear Bold:
    # Both h-stems and v-stems = REG_STEM + 2×DELTA  ≈ 33 + 40 = 73
    bold_master.horizontalStems = [33 + 2 * DELTA]
    bold_master.verticalStems   = [33 + 2 * DELTA]

    for cp in reg_master.customParameters:
        bold_master.customParameters.append(GSCustomParameter(cp.name, cp.value))

    bold_master.customParameters.append(
        GSCustomParameter("Axis Location", [{"Axis": "Weight", "Location": 700}])
    )

    bold_id = bold_master.id
    print(f"Bold master id: {bold_id}")
    font.masters.append(bold_master)

    # ── 3. Copy kerning ────────────────────────────────────────────────
    reg_kerning          = font.kerning.get(reg_id, {})
    font.kerning[bold_id] = copy.deepcopy(reg_kerning)
    print(f"Copied kerning: {len(reg_kerning)} first-level entries")

    # ── 4. Build Bold layers for every glyph ──────────────────────────
    no_outline = with_paths = composite = mixed = 0

    for glyph in font.glyphs:
        reg_layer = glyph.layers[reg_id]
        if reg_layer is None:
            continue

        has_paths = len(reg_layer.paths)      > 0
        has_comps = len(reg_layer.components) > 0

        bl = make_bold_layer(reg_layer, bold_id)
        glyph.layers.append(bl)

        if reg_layer.smartComponentPoleMapping:
            bl.smartComponentPoleMapping = dict(reg_layer.smartComponentPoleMapping)

        if   has_paths and has_comps: mixed      += 1
        elif has_paths:               with_paths += 1
        elif has_comps:               composite  += 1
        else:                         no_outline += 1

    print(f"Processed glyphs: {with_paths} paths-only  "
          f"{composite} composite  {mixed} mixed  {no_outline} empty")

    # ── 4b. Bold alternate layers for smart-component glyphs ──────────
    _date_re      = re.compile(r'^Regular [A-Z][a-z]{2} ')
    smart_alt_added = 0

    for glyph in font.glyphs:
        for alt_layer in list(glyph.layers):
            if alt_layer.associatedMasterId != reg_id:
                continue
            if alt_layer.layerId == reg_id:
                continue
            if _date_re.match(alt_layer.name or ''):
                continue

            bold_alt_exists = any(
                la.associatedMasterId == bold_id and la.name == alt_layer.name
                for la in glyph.layers
            )
            if bold_alt_exists:
                continue

            new_id                = str(uuid.uuid4()).upper()
            bl                    = make_bold_layer(alt_layer, new_id)
            bl.layerId            = new_id
            bl.associatedMasterId = bold_id
            bl.name               = alt_layer.name
            glyph.layers.append(bl)
            if alt_layer.smartComponentPoleMapping:
                bl.smartComponentPoleMapping = dict(alt_layer.smartComponentPoleMapping)
            smart_alt_added += 1

    print(f"Added Bold alternates for smart-component layers: {smart_alt_added}")

    # ── 5. Variable-font axis metadata ────────────────────────────────
    font.customParameters.append(
        GSCustomParameter("Axes", [{"Name": "Weight", "Tag": "wght"}])
    )
    font.customParameters.append(
        GSCustomParameter("Variable Font Origin", "Regular")
    )

    # ── 6. Named instances ────────────────────────────────────────────
    font.instances.clear()
    for name, wv in [("Regular", 400), ("Medium", 500), ("SemiBold", 600), ("Bold", 700)]:
        inst             = GSInstance()
        inst.name        = name
        inst.weightValue = wv
        inst.customParameters.append(
            GSCustomParameter("Axis Location", [{"Axis": "Weight", "Location": wv}])
        )
        font.instances.append(inst)

    # ── 7. Save ────────────────────────────────────────────────────────
    print(f"Saving  {OUTPUT}")
    font.save(OUTPUT)
    print("Done.")


if __name__ == "__main__":
    main()
