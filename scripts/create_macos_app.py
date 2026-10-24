"""Build a local .app launcher tied to this checkout and its uv environment."""
from __future__ import annotations

import os
import plistlib
import shlex
import struct
import sys
from pathlib import Path


def build(root: Path | None = None) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("このランチャーはmacOS専用です。")
    root = root or Path(__file__).resolve().parents[1]
    app = root / "output" / "desktop" / "RomRaider Log Analyzer.app"
    contents = app / "Contents"
    executable = contents / "MacOS" / "RomRaiderLogAnalyzer"
    resources = contents / "Resources"
    resources.mkdir(parents=True, exist_ok=True)
    executable.parent.mkdir(parents=True, exist_ok=True)
    python = root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError("先に uv sync --extra desktop を実行してください。")
    # Quote every path, including checkouts containing spaces or shell characters.
    executable.write_text("#!/bin/sh\n"
                          + f"cd {shlex.quote(str(root))} || exit 1\n"
                          + "mkdir -p .cache/desktop\n"
                          + f"exec {shlex.quote(str(python))} -m src.desktop \"$@\" >> .cache/desktop/launch.log 2>&1\n",
                          encoding="utf-8")
    executable.chmod(0o755)
    info = {"CFBundleName": "RomRaider Log Analyzer",
            "CFBundleDisplayName": "RomRaider Log Analyzer",
            "CFBundleIdentifier": "local.sg9.romraider-analyzer",
            "CFBundleExecutable": executable.name,
            "CFBundlePackageType": "APPL", "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "1", "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication"}
    make_icon(root, resources)
    info["CFBundleIconFile"] = "AppIcon"
    with (contents / "Info.plist").open("wb") as handle:
        plistlib.dump(info, handle)
    return app


def make_icon(root, resources):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    iconset = root / ".cache" / "desktop" / "AppIcon.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    for scale in (16, 32, 128, 256, 512):
        for retina in (1, 2):
            size = scale * retina
            image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
            image.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(image)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.scale(size/512, size/512)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor("#203c5a"))
            painter.drawRoundedRect(QtCore.QRectF(12,12,488,488),105,105)
            painter.setPen(QtGui.QPen(QtGui.QColor("#4f6a82"),2))
            for y in (140,220,300,380):
                painter.drawLine(75,y,440,y)
            for x in (100,180,260,340,420):
                painter.drawLine(x,100,x,415)
            path = QtGui.QPainterPath(QtCore.QPointF(75,335))
            for x,y in ((135,325),(180,260),(210,295),(245,180),(285,215),(345,120),(390,165),(440,115)):
                path.lineTo(x,y)
            painter.setPen(QtGui.QPen(QtGui.QColor("#8bbdff"),15,QtCore.Qt.PenStyle.SolidLine,
                                     QtCore.Qt.PenCapStyle.RoundCap,QtCore.Qt.PenJoinStyle.RoundJoin))
            painter.drawPath(path)
            painter.end()
            suffix = "@2x" if retina==2 else ""
            image.save(str(iconset / f"icon_{scale}x{scale}{suffix}.png"))
    # PNG-backed ICNS chunks avoid invoking system image conversion services.
    chunks = []
    for kind, filename in ((b"icp4", "icon_16x16.png"), (b"icp5", "icon_32x32.png"),
                            (b"icp6", "icon_32x32@2x.png"), (b"ic07", "icon_128x128.png"),
                            (b"ic08", "icon_256x256.png"), (b"ic09", "icon_512x512.png"),
                            (b"ic10", "icon_512x512@2x.png")):
        data = (iconset / filename).read_bytes()
        chunks.append(kind + struct.pack(">I", len(data)+8) + data)
    data = b"".join(chunks)
    (resources / "AppIcon.icns").write_bytes(b"icns" + struct.pack(">I",len(data)+8) + data)
    del app


if __name__ == "__main__":
    print(build())
