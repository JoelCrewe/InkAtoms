"""Command line: inkatoms structure.xyz -o figure.pdf --view -y --style cpk"""

from __future__ import annotations

import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(prog="inkatoms", description="Draw an atomic structure to a PDF/SVG/PNG.")
    p.add_argument("file", help="any structure file ASE can read (xyz, extxyz, cif, POSCAR, cube, ...)")
    p.add_argument("-o", "--output", help="output file (default: <file>.pdf)")
    p.add_argument("-i", "--index", default="-1", help="frame to draw for trajectories (default -1, the last)")
    p.add_argument("--view", default="auto", help="out-of-page direction: x, -y, z, ... or auto (default)")
    p.add_argument("--rotate", default=None, help="extra rotation, e.g. 20x,-30y")
    p.add_argument("--style", default="jmol", help="jmol, cpk, vesta, colourblind, greyscale, bw, dark")
    p.add_argument("--mode", default="ballstick", choices=["ballstick", "spacefill", "sticks"])
    p.add_argument("--shading", default=None, choices=["flat", "glossy", "soft"])
    p.add_argument("--no-depth-cue", action="store_true", help="no fading or shrinking with depth")
    p.add_argument("--perspective", type=float, default=None, help="camera distance in Å")
    p.add_argument("--cell", action="store_true", help="draw the unit cell")
    p.add_argument("--repeat", default=None, help="tile the cell, e.g. 3,3,1")
    p.add_argument("--boundary", action="store_true", help="repeat atoms on cell faces (VESTA style)")
    p.add_argument("--bond-orders", action="store_true", help="draw double/triple bonds")
    p.add_argument("--polyhedra", action="store_true", help="draw metal-centred polyhedra")
    p.add_argument("--color-by", default=None, help="z, coordination or a per-atom array name")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--highlight", type=int, nargs="*", default=None, help="atom indices to highlight")
    p.add_argument("--labels", default=None, choices=["symbol", "id", "index", "value"])
    p.add_argument("--legend", action="store_true")
    p.add_argument("--scale", type=float, default=0.3, help="inches per Å (default 0.3)")
    p.add_argument("--dpi", type=int, default=300, help="for raster outputs")
    p.add_argument("--transparent", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        from ase.io import read
    except ImportError:
        sys.exit("inkatoms needs ASE to read files: pip install ase")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import extras, select
    from .render import structure_figure
    from .structure import as_structure

    index = int(args.index) if args.index.lstrip("-").isdigit() else args.index
    atoms = read(args.file, index=index)
    s = as_structure(atoms)
    if args.repeat:
        s = select.tile(s, tuple(int(x) for x in args.repeat.split(",")), centred=False)
    kw = dict(depth_cue=not args.no_depth_cue, view=args.view, rotate=args.rotate, style=args.style, mode=args.mode,
              perspective=args.perspective, cell=args.cell, boundary=args.boundary,
              bond_orders=args.bond_orders, polyhedra=args.polyhedra or None,
              color_by=args.color_by, cmap=args.cmap, highlight=args.highlight, labels=args.labels)
    if args.shading:
        kw["shading"] = args.shading
    try:
        fig, ax, art = structure_figure(s, inches_per_angstrom=args.scale, **kw)
    except ValueError as err:
        sys.exit(str(err))
    if args.legend:
        extras.legend(ax, art, loc="upper left", fontsize=7)
    if args.color_by:
        extras.colorbar(art, label=args.color_by)
    out = args.output or args.file.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(out, dpi=args.dpi, transparent=args.transparent)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
