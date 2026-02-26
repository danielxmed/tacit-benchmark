# tests/core/test_renderer.py
import pytest
from pathlib import Path
import tempfile


def test_style_constants_exist():
    from tacit.core.renderer import STYLE
    assert "background" in STYLE
    assert "line_width" in STYLE
    assert "colors" in STYLE
    assert isinstance(STYLE["colors"], list)


def test_create_svg_canvas():
    from tacit.core.renderer import create_canvas
    canvas = create_canvas(width=512, height=512)
    svg_str = canvas.tostring()
    assert "svg" in svg_str
    assert '512' in svg_str


def test_draw_rect():
    from tacit.core.renderer import create_canvas, draw_rect
    canvas = create_canvas(256, 256)
    draw_rect(canvas, x=10, y=10, width=50, height=50, fill="#FF0000")
    svg_str = canvas.tostring()
    assert "rect" in svg_str
    assert "FF0000" in svg_str


def test_draw_circle():
    from tacit.core.renderer import create_canvas, draw_circle
    canvas = create_canvas(256, 256)
    draw_circle(canvas, cx=100, cy=100, r=25, fill="#00FF00")
    svg_str = canvas.tostring()
    assert "circle" in svg_str


def test_draw_line():
    from tacit.core.renderer import create_canvas, draw_line
    canvas = create_canvas(256, 256)
    draw_line(canvas, x1=0, y1=0, x2=100, y2=100)
    svg_str = canvas.tostring()
    assert "line" in svg_str


def test_draw_path():
    from tacit.core.renderer import create_canvas, draw_path
    canvas = create_canvas(256, 256)
    draw_path(canvas, d="M 10 10 L 50 50 L 90 10 Z", fill="none", stroke="#000")
    svg_str = canvas.tostring()
    assert "path" in svg_str


def test_draw_text():
    from tacit.core.renderer import create_canvas, draw_text
    canvas = create_canvas(256, 256)
    draw_text(canvas, x=50, y=50, text="A", font_size=14)
    svg_str = canvas.tostring()
    assert ">A<" in svg_str


def test_svg_to_string():
    from tacit.core.renderer import create_canvas, svg_to_string
    canvas = create_canvas(100, 100)
    s = svg_to_string(canvas)
    assert isinstance(s, str)
    assert s.startswith("<?xml") or s.startswith("<svg")


def test_svg_to_png():
    from tacit.core.renderer import create_canvas, svg_to_png
    canvas = create_canvas(100, 100)
    png_bytes = svg_to_png(canvas, width=256)
    # PNG magic bytes
    assert png_bytes[:4] == b'\x89PNG'


def test_svg_to_png_multiple_resolutions():
    from tacit.core.renderer import create_canvas, svg_to_png_multi
    canvas = create_canvas(100, 100)
    result = svg_to_png_multi(canvas, widths=[256, 512])
    assert 256 in result
    assert 512 in result
    assert result[256][:4] == b'\x89PNG'
    assert result[512][:4] == b'\x89PNG'


def test_save_svg_to_file():
    from tacit.core.renderer import create_canvas, draw_rect, save_svg
    canvas = create_canvas(100, 100)
    draw_rect(canvas, 0, 0, 100, 100, fill="#FFF")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.svg"
        save_svg(canvas, path)
        assert path.exists()
        content = path.read_text()
        assert "svg" in content


def test_save_png_to_file():
    from tacit.core.renderer import create_canvas, save_png
    canvas = create_canvas(100, 100)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.png"
        save_png(canvas, path, width=256)
        assert path.exists()
        assert path.read_bytes()[:4] == b'\x89PNG'
