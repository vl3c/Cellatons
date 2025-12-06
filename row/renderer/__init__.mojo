"""Renderer module for row cellular automata visualization.

Provides fullscreen pygame rendering with:
- Zero-copy numpy view over Mojo grid data
- Hardware-accelerated double-buffered display
- Real-time progress and FPS overlay

Usage:
    pixi run mojo row/renderer/main.mojo

Components:
- base.mojo: RendererConfig and display initialization
- python_module.mojo: Mojo-side renderer wrapper
- viewer.py: Python-side rendering logic (numpy + pygame)
- main.mojo: Live viewer application
"""

from .python_module import PythonModuleRenderer
