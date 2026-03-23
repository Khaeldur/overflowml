"""Auto-clicker agent — launches TUI, clicks every button, tracks output, reports issues.

Run with ComfyUI venv: .venv/Scripts/python.exe -m pytest tests/test_tui_autoclick.py -v
Requires: pip install overflowml[ui] pytest-asyncio
"""

import sys
import pytest

sys.path.insert(0, ".")

try:
    from overflowml.tui import _get_app_class
    from textual.widgets import Input, Button
    OverflowMLApp = _get_app_class()
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False
    OverflowMLApp = None

pytestmark = [
    pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed"),
    pytest.mark.asyncio,
]


@pytest.fixture
def app():
    return OverflowMLApp()


async def _click_button(pilot, app, btn_id):
    """Scroll button into view then click it."""
    btn = app.query_one(f"#{btn_id}", Button)
    btn.scroll_visible()
    await pilot.pause()
    await pilot.click(f"#{btn_id}")
    await pilot.pause()


async def test_app_starts(app):
    async with app.run_test(size=(120, 50)) as pilot:
        assert app.title == "OverflowML"


async def test_click_detect(app):
    async with app.run_test(size=(120, 50)) as pilot:
        await _click_button(pilot, app, "btn-detect")


async def test_click_doctor(app):
    async with app.run_test(size=(120, 50)) as pilot:
        await _click_button(pilot, app, "btn-doctor")


async def test_click_plan(app):
    async with app.run_test(size=(120, 50)) as pilot:
        await _click_button(pilot, app, "btn-plan")


async def test_click_compare(app):
    async with app.run_test(size=(120, 50)) as pilot:
        await _click_button(pilot, app, "btn-compare")


async def test_click_canrun(app):
    async with app.run_test(size=(120, 50)) as pilot:
        await _click_button(pilot, app, "btn-canrun")


async def test_click_benchmark(app):
    async with app.run_test(size=(120, 50)) as pilot:
        await _click_button(pilot, app, "btn-benchmark")


async def test_click_cache(app):
    async with app.run_test(size=(120, 50)) as pilot:
        await _click_button(pilot, app, "btn-cache")


async def test_click_vram(app):
    async with app.run_test(size=(120, 50)) as pilot:
        await _click_button(pilot, app, "btn-vram")


async def test_click_all_buttons_sequence(app):
    """The full autoclick run — every button in order."""
    async with app.run_test(size=(120, 50)) as pilot:
        for btn_id in ["btn-detect", "btn-doctor", "btn-plan", "btn-compare",
                       "btn-canrun", "btn-benchmark", "btn-cache", "btn-vram"]:
            await _click_button(pilot, app, btn_id)


async def test_custom_input_plan(app):
    async with app.run_test(size=(120, 50)) as pilot:
        app.query_one("#model-input", Input).value = "70"
        await _click_button(pilot, app, "btn-plan")


async def test_custom_input_compare(app):
    async with app.run_test(size=(120, 50)) as pilot:
        app.query_one("#model-input", Input).value = "140"
        await _click_button(pilot, app, "btn-compare")


async def test_keyboard_shortcuts(app):
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("o")
        await pilot.pause()
        await pilot.press("p")
        await pilot.pause()
        await pilot.press("b")
        await pilot.pause()
