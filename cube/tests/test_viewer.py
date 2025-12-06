"""Tests for cube.renderer.viewer (Python)."""

import numpy as np
import unittest
from unittest.mock import Mock

import viewer  # module import path set by run_tests


class TestCreateGridView(unittest.TestCase):
    def test_shape_and_zero_copy(self):
        depth, height, stride = 4, 3, 5
        data = np.zeros(depth * height * stride, dtype=np.uint8)
        ptr = data.ctypes.data
        
        view = viewer.create_grid_view(ptr, stride, height, stride, depth)
        self.assertEqual(view.shape, (depth, height, stride))
        
        view[1, 1, 1] = 9
        self.assertEqual(data[(1 * height + 1) * stride + 1], 9)


class TestRenderFrame(unittest.TestCase):
    def setUp(self):
        viewer.reset_state()
    
    def test_render_frame_runs(self):
        depth, height, stride = 3, 3, 3
        grid = np.zeros((depth, height, stride), dtype=np.uint8)
        grid[1, 1, 1] = 1
        
        mock_draw = Mock()
        mock_draw.aaline = Mock()
        
        mock_font = Mock()
        mock_font.render.return_value = Mock(get_rect=Mock(return_value=Mock()))
        
        mock_pygame = Mock()
        mock_pygame.Surface.return_value = Mock()
        mock_pygame.font.init = Mock()
        mock_pygame.font.SysFont.return_value = mock_font
        mock_pygame.surfarray.blit_array = Mock()
        mock_pygame.draw = mock_draw
        mock_pygame.SRCALPHA = 0
        
        mock_screen = Mock()
        mock_screen.fill = Mock()
        mock_screen.blit = Mock()
        
        # Should not raise
        viewer.render_frame(
            mock_pygame,
            mock_screen,
            grid,
            320,
            240,
            generation=0,
            fps=60,
            gen_time_ms=1.2,
        )
        
        mock_pygame.surfarray.blit_array.assert_called()
        mock_screen.blit.assert_called()

    def test_render_bounds_not_touching_edges(self):
        """Ensure rendered cube content stays within viewport margins."""
        depth, height, stride = 3, 3, 3
        grid = np.zeros((depth, height, stride), dtype=np.uint8)
        grid[1, 1, 1] = 1
        
        mock_draw = Mock()
        mock_draw.aaline = Mock()
        
        mock_font = Mock()
        mock_font.render.return_value = Mock(get_rect=Mock(return_value=Mock()))
        
        mock_pygame = Mock()
        mock_pygame.Surface.return_value = Mock()
        mock_pygame.font.init = Mock()
        mock_pygame.font.SysFont.return_value = mock_font
        mock_pygame.surfarray.blit_array = Mock()
        mock_pygame.draw = mock_draw
        mock_pygame.SRCALPHA = 0
        
        mock_screen = Mock()
        mock_screen.fill = Mock()
        mock_screen.blit = Mock()
        
        viewer.reset_state()
        viewer.render_frame(
            mock_pygame,
            mock_screen,
            grid,
            400,
            400,
            generation=0,
            fps=60,
            gen_time_ms=1.2,
            paused=True,
            width_logical=3,
            height_logical=3,
            depth_logical=3,
        )
        
        buf = viewer._state.rgb_buffer
        mask = buf.sum(axis=2) > 0
        self.assertTrue(mask.any())
        ys, xs = np.nonzero(mask)
        margin = 10
        self.assertGreater(ys.min(), margin)
        self.assertGreater(xs.min(), margin)
        self.assertLess(ys.max(), buf.shape[0] - margin)
        self.assertLess(xs.max(), buf.shape[1] - margin)


if __name__ == "__main__":
    unittest.main()


