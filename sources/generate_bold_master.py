#!/usr/bin/env python3
"""
generate_bold_master.py
───────────────────────
Adds a Bold master to Poiret One, producing a two-master .glyphs source
ready for variable-font compilation.

Design targets
  Regular (wght=400)  stems ≈  33 units  — Art Deco hairline
  Bold    (wght=700)  stems ≈  73 units  — strong display weight

Algorithm: per-node miter-offset expansion (DELTA = 20 units per side).
  • CCW (outer fill) paths → nodes move outward   → shape expands
  • CW  (inner counter) paths → nodes move inward → counter contracts
  Both together produce visually thicker strokes.

After displacement, nodes within SNAP units of a vertical alignment
metric (y = 0, 450, 750, −208, 962) are snapped back to that metric
so baseline and cap-height never drift.
"""

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
SOURCE  = "/home/user/poirettwo/sources/PoiretOne.glyphs"
OUTPUT  = "/home/user/poirettwo/sources/PoiretOneVF.glyphs"

DELTA   = 20        # expansion per side, font units
SNAP    = 3         # snap bold-y to metric when original is within SNAP of it

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
    """
    n   = len(positions)
    p   = positions[i]
    pp  = positions[(i - 1) % n]
    pn  = positions[(i + 1) % n]

    # Incoming and outgoing segment vectors
    dx1, dy1 = p[0] - pp[0], p[1] - pp[1]
    dx2, dy2 = pn[0] - p[0], pn[1] - p[1]
    m1 = math.hypot(dx1, dy1)
    m2 = math.hypot(dx2, dy2)

    # Handle degenerate (zero-length) segments
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

    # Bisector of the two normals
    bx, by = n1x + n2x, n1y + n2y
    bm = math.hypot(bx, by)
    if bm < 1e-9:
        # 180° u-turn → use n1 straight
        return delta * n1x, delta * n1y
    bx /= bm
    by /= bm

    # proj = cos(half-angle between the two normals)
    proj = bx * n1x + by * n1y
    if abs(proj) < 1e-9:
        return delta * n1x, delta * n1y

    scale = delta / proj
    # Apply miter limit to avoid extreme spikes at very acute corners
    if abs(scale) > miter_limit * abs(delta):
        scale = math.copysign(miter_limit * abs(delta), scale)

    return scale * bx, scale * by


def snap_y_to_metric(orig_y: float, bold_y: float) -> float:
    """If orig_y is within SNAP of an alignment metric, snap bold_y to that metric."""
    for m in METRICS:
        if abs(orig_y - m) <= SNAP:
            return float(m)
    return bold_y


def fix_colinear_metric_nodes(
    positions: list[tuple[float, float]],
    new_positions: list[tuple[float, float]],
    delta: float,
) -> list[tuple[float, float]]:
    """
    Post-processing correction for co-linear y-metric nodes.

    Scenario: a node sits exactly on a vertical alignment metric (y = 0,
    450, 750 …) AND both its immediate neighbours are at the *same* y
    (i.e. all three lie on a horizontal edge).  In that case the miter
    algorithm produces a purely-vertical displacement which snap_y_to_metric
    then zeros out, leaving the node unmoved.

    This matters at stem-to-bowl transitions: e.g., the bottom-right corner
    of the 'B' stem at (111, 0) has leftward and rightward neighbours both at
    y = 0, so the stem fails to widen there.

    Fix: for any such "stuck" node, move it laterally (toward the path
    centroid) by DELTA so that the stem attachment expands symmetrically.
    """
    n = len(positions)
    centroid_x = sum(p[0] for p in positions) / n
    fixed = list(new_positions)

    for i in range(n):
        ox, oy = positions[i]
        nx, ny = new_positions[i]

        # Only nodes that ended up unmoved
        if abs(nx - ox) > 0.5 or abs(ny - oy) > 0.5:
            continue

        # Only nodes originally on a vertical alignment metric
        if not any(abs(oy - m) <= SNAP for m in METRICS):
            continue

        # Co-linearity check: both neighbours within SNAP of the same y
        p_prev = positions[(i - 1) % n]
        p_next = positions[(i + 1) % n]
        if abs(p_prev[1] - oy) > SNAP or abs(p_next[1] - oy) > SNAP:
            continue

        # Apply centroid-directed lateral correction
        if ox < centroid_x - 1:
            fixed[i] = (ox + delta, oy)
        elif ox > centroid_x + 1:
            fixed[i] = (ox - delta, oy)

    return fixed


# ── Layer construction ─────────────────────────────────────────────────

def make_bold_path(orig_path: GSPath) -> GSPath:
    """Return a new GSPath with each node displaced by DELTA via miter offset."""
    positions = [(nd.position.x, nd.position.y) for nd in orig_path.nodes]
    types     = [nd.type   for nd in orig_path.nodes]
    smooths   = [nd.smooth for nd in orig_path.nodes]

    new_path        = GSPath()
    new_path.closed = orig_path.closed
    new_nodes       = []

    for i, (ox, oy) in enumerate(positions):
        ddx, ddy = miter_displacement(positions, i, DELTA)
        nx = ox + ddx
        ny = snap_y_to_metric(oy, oy + ddy)
        node = GSNode((nx, ny), type=types[i], smooth=smooths[i])
        new_nodes.append(node)

    new_path.nodes = new_nodes
    return new_path


def make_bold_layer(reg_layer: GSLayer, bold_master_id: str) -> GSLayer:
    """
    Build the Bold GSLayer for one glyph.
    Paths are expanded; components, anchors, and advance width are copied.
    """
    bl = GSLayer()
    bl.layerId          = bold_master_id
    bl.associatedMasterId = bold_master_id
    bl.name             = "Bold"
    bl.width            = reg_layer.width   # keep advance width; SBs narrow naturally

    # Expand drawn outlines
    for path in reg_layer.paths:
        bl.paths.append(make_bold_path(path))

    # Component references are unchanged — they'll use their own Bold layers
    for comp in reg_layer.components:
        bl.components.append(comp.clone())

    # Anchors: y-coords on metrics stay on metrics; others stay put for now
    for anch in reg_layer.anchors:
        a          = GSAnchor()
        a.name     = anch.name
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

    # ── 1. Stamp explicit axis location on the Regular master ─────────
    # glyphsLib's font_uses_axis_locations() requires ALL masters to have
    # an "Axis Location" custom parameter; only then does it use the reliable
    # explicit-location branch.  Without it, instance-based heuristics fire
    # and collapse the axis to a single point.
    reg_master.customParameters.append(
        GSCustomParameter("Axis Location", [{"Axis": "Weight", "Location": 400}])
    )

    # ── 1. Create Bold master ──────────────────────────────────────────
    bold_master = GSFontMaster()
    bold_master.name        = "Bold"
    bold_master.weightValue = 700
    bold_master.widthValue  = reg_master.widthValue

    # Copy vertical metrics
    bold_master.ascender  = reg_master.ascender
    bold_master.descender = reg_master.descender
    bold_master.capHeight = reg_master.capHeight
    bold_master.xHeight   = reg_master.xHeight

    # Copy alignment zones (vertical metrics don't change)
    bold_master.alignmentZones = list(reg_master.alignmentZones)

    # Stem values for Bold (hinting and spacing guidance)
    # Regular h-stems: [29, 28], v-stems: [33, 29]
    # Bold expands by 2×DELTA each = +40 units per stem
    bold_master.horizontalStems = [s + 2 * DELTA for s in reg_master.horizontalStems]
    bold_master.verticalStems   = [s + 2 * DELTA for s in reg_master.verticalStems]

    # Copy master-level metric custom parameters verbatim
    for cp in reg_master.customParameters:
        bold_master.customParameters.append(
            GSCustomParameter(cp.name, cp.value)
        )

    # Explicit axis location for Bold master
    bold_master.customParameters.append(
        GSCustomParameter("Axis Location", [{"Axis": "Weight", "Location": 700}])
    )

    bold_id = bold_master.id
    print(f"Bold master id: {bold_id}")
    font.masters.append(bold_master)

    # ── 2. Add Bold kerning (copy from Regular as starting point) ──────
    reg_kerning = font.kerning.get(reg_id, {})
    # Deep-copy the dict structure
    import copy
    font.kerning[bold_id] = copy.deepcopy(reg_kerning)
    print(f"Copied kerning: {len(reg_kerning)} first-level entries")

    # ── 3. Build Bold layers for every glyph ───────────────────────────
    no_outline = 0
    with_paths = 0
    composite  = 0
    mixed      = 0

    for glyph in font.glyphs:
        reg_layer = glyph.layers[reg_id]
        if reg_layer is None:
            continue

        has_paths  = len(reg_layer.paths)      > 0
        has_comps  = len(reg_layer.components) > 0

        bl = make_bold_layer(reg_layer, bold_id)
        glyph.layers.append(bl)
        # Copy smart-component pole mapping (needs parent to be set first)
        if reg_layer.smartComponentPoleMapping:
            bl.smartComponentPoleMapping = dict(reg_layer.smartComponentPoleMapping)

        if has_paths and has_comps:
            mixed += 1
        elif has_paths:
            with_paths += 1
        elif has_comps:
            composite += 1
        else:
            no_outline += 1

    print(f"Processed glyphs: {with_paths} paths-only  "
          f"{composite} composite  {mixed} mixed  {no_outline} empty")

    # ── 3b. Add Bold alternate layers for smart-component part glyphs ──
    # Each _part.* glyph has a "Short"/"short" alternate layer (a
    # Glyphs smart-component master).  glyphsLib requires a matching Bold
    # alternate layer when building a variable font.
    # We skip date-stamped backup layers ("Regular Jul …").
    _date_re = re.compile(r'^Regular [A-Z][a-z]{2} ')
    smart_alt_added = 0

    for glyph in font.glyphs:
        for alt_layer in list(glyph.layers):   # snapshot to avoid mutation issues
            # Only handle Regular-master alternate layers that look like design variants
            if alt_layer.associatedMasterId != reg_id:
                continue
            if alt_layer.layerId == reg_id:
                continue
            if _date_re.match(alt_layer.name or ''):
                continue

            # Check whether a matching Bold alternate already exists
            bold_alt_exists = any(
                la.associatedMasterId == bold_id
                and la.name == alt_layer.name
                for la in glyph.layers
            )
            if bold_alt_exists:
                continue

            # Create Bold alternate layer
            new_id = str(uuid.uuid4()).upper()
            bl = make_bold_layer(alt_layer, new_id)   # reuse our helper
            # Override the IDs: alternates share associatedMasterId with Bold
            bl.layerId            = new_id
            bl.associatedMasterId = bold_id
            bl.name               = alt_layer.name    # keep "Short" / "short"
            # Copy smart-component pole mapping from the Regular alternate
            glyph.layers.append(bl)
            # Set pole mapping after appending (needs parent to be set first)
            if alt_layer.smartComponentPoleMapping:
                bl.smartComponentPoleMapping = dict(alt_layer.smartComponentPoleMapping)
            smart_alt_added += 1

    print(f"Added Bold alternates for smart-component layers: {smart_alt_added}")

    # ── 4. Add variable-font axis metadata ────────────────────────────
    font.customParameters.append(
        GSCustomParameter("Axes", [{"Name": "Weight", "Tag": "wght"}])
    )
    font.customParameters.append(
        GSCustomParameter("Variable Font Origin", "Regular")
    )

    # ── 5. Update instances for named styles ──────────────────────────
    # Use pure weightValue-based locations — no instanceInterpolations,
    # which would engage manual-interpolation mode and confuse glyphsLib's
    # axis-range detection.
    font.instances.clear()

    for name, wv in [("Regular", 400), ("Medium", 500), ("SemiBold", 600), ("Bold", 700)]:
        inst             = GSInstance()
        inst.name        = name
        inst.weightValue = wv
        # Explicit axis location so glyphsLib can read user-space positions
        inst.customParameters.append(
            GSCustomParameter("Axis Location", [{"Axis": "Weight", "Location": wv}])
        )
        font.instances.append(inst)

    # ── 6. Save ────────────────────────────────────────────────────────
    print(f"Saving  {OUTPUT}")
    font.save(OUTPUT)
    print("Done.")


if __name__ == "__main__":
    main()
