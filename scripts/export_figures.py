#!/usr/bin/env python3
"""Export SVG diagrams to PDF and high-resolution PNG for paper submission.

Usage:
    # Install dependencies first:
    pip install cairosvg

    # Export all figures:
    python scripts/export_figures.py

    # Export specific format only:
    python scripts/export_figures.py --format pdf
    python scripts/export_figures.py --format png

    # Custom DPI for PNG:
    python scripts/export_figures.py --format png --dpi 600

Output:
    figures/pdf/   — Vector PDF files (for LaTeX)
    figures/png/   — High-resolution PNG files (for Word/PowerPoint)
"""

import argparse
import sys
from pathlib import Path

try:
    import cairosvg
except ImportError:
    print("Error: cairosvg is required. Install it with:")
    print("  pip install cairosvg")
    print()
    print("On some systems you may also need Cairo libraries:")
    print("  Ubuntu/Debian: sudo apt install libcairo2-dev")
    print("  macOS:         brew install cairo")
    print("  Windows:       pip install cairosvg (usually works directly)")
    sys.exit(1)


# SVG files to export
SVG_FILES = [
    # Architecture diagrams
    "system_overview.svg",
    "architecture.svg",
    "caam_module.svg",
    "multi_task_loss.svg",
    "training_pipeline.svg",
    "constellations.svg",
    "signal_processing.svg",
    # Result figures
    "results_training_curves.svg",
    "results_osnr_scatter.svg",
    "results_confusion_matrix.svg",
    "results_error_distribution.svg",
    "results_osnr_per_modulation.svg",
    "results_osnr_vs_error.svg",
]


def export_to_pdf(svg_path: Path, output_dir: Path):
    """Convert SVG to PDF (vector — ideal for LaTeX)."""
    output = output_dir / svg_path.with_suffix(".pdf").name
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(output))
    return output


def export_to_png(svg_path: Path, output_dir: Path, dpi: int = 300):
    """Convert SVG to high-resolution PNG (for Word/PowerPoint)."""
    # cairosvg uses 96 DPI by default, scale accordingly
    scale = dpi / 96.0
    output = output_dir / svg_path.with_suffix(".png").name
    cairosvg.svg2png(url=str(svg_path), write_to=str(output), scale=scale)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Export SVG diagrams to PDF/PNG for paper submission"
    )
    parser.add_argument(
        "--format", choices=["pdf", "png", "both"], default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for PNG export (default: 300, use 600 for print)",
    )
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).resolve().parent.parent
    figures_dir = project_root / "figures"
    pdf_dir = figures_dir / "pdf"
    png_dir = figures_dir / "png"

    # Create output directories
    if args.format in ("pdf", "both"):
        pdf_dir.mkdir(exist_ok=True)
    if args.format in ("png", "both"):
        png_dir.mkdir(exist_ok=True)

    print(f"Exporting {len(SVG_FILES)} diagrams...\n")

    for svg_name in SVG_FILES:
        svg_path = figures_dir / svg_name
        if not svg_path.exists():
            print(f"  [SKIP] {svg_name} — not found")
            continue

        if args.format in ("pdf", "both"):
            out = export_to_pdf(svg_path, pdf_dir)
            print(f"  [PDF] {out.relative_to(project_root)}")

        if args.format in ("png", "both"):
            out = export_to_png(svg_path, png_dir, args.dpi)
            print(f"  [PNG] {out.relative_to(project_root)} ({args.dpi} DPI)")

    print(f"\nDone! Files saved to:")
    if args.format in ("pdf", "both"):
        print(f"  PDF: {pdf_dir.relative_to(project_root)}/")
    if args.format in ("png", "both"):
        print(f"  PNG: {png_dir.relative_to(project_root)}/")

    print("\n--- LaTeX Usage ---")
    print(r"""
\usepackage{graphicx}

% For PDF figures (recommended):
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/pdf/architecture.pdf}
    \caption{MT-OPMNet architecture.}
    \label{fig:architecture}
\end{figure}
""")


if __name__ == "__main__":
    main()
