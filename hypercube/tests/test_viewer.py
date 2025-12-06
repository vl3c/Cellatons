"""Tests for hypercube.renderer.viewer (Python)."""

import numpy as np
import unittest
from unittest.mock import Mock

import viewer  # module import path set by run_tests


class TestCreateGridView(unittest.TestCase):
    def test_shape_and_zero_copy(self):
        w_dim, depth, height, stride = 2, 3, 3, 4
        data = np.zeros(w_dim * depth * height * stride, dtype=np.uint8)
        ptr = data.ctypes.data
        
        view = viewer.create_grid_view(ptr, stride, height, stride, depth, w_dim)
        self.assertEqual(view.shape, (w_dim, depth, height, stride))
        
        view[1, 1, 1, 1] = 7
        idx = ((1 * depth + 1) * height + 1) * stride + 1
        self.assertEqual(data[idx], 7)


class TestRenderFrame(unittest.TestCase):
    def setUp(self):
        viewer.reset_state()
    
    def _make_mocks(self):
        mock_draw = Mock()
        mock_draw.aaline = Mock()
        
        mock_font = Mock()
        surface_mock = Mock()
        surface_mock.get_rect = Mock(return_value=Mock())
        surface_mock.get_height = Mock(return_value=16)
        mock_font.render.return_value = surface_mock
        
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
        return mock_pygame, mock_screen
    
    def test_render_frame_runs(self):
        w_dim, depth, height, stride = 2, 3, 3, 3
        grid = np.zeros((w_dim, depth, height, stride), dtype=np.uint8)
        grid[0, 1, 1, 1] = 1
        
        mock_pygame, mock_screen = self._make_mocks()
        
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
            paused=True,
            render_mode=0,  # Slice
            slice_index=0,
            width_logical=3,
            height_logical=3,
            depth_logical=3,
            w_dim=w_dim,
        )
        
        mock_pygame.surfarray.blit_array.assert_called()
        mock_screen.blit.assert_called()
    
    def test_render_bounds_not_touching_edges(self):
        """Ensure rendered projection stays within viewport margins."""
        w_dim, depth, height, stride = 2, 3, 3, 3
        grid = np.zeros((w_dim, depth, height, stride), dtype=np.uint8)
        grid[0, 1, 1, 1] = 1
        
        mock_pygame, mock_screen = self._make_mocks()
        
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
            render_mode=0,  # Slice
            slice_index=0,
            width_logical=3,
            height_logical=3,
            depth_logical=3,
            w_dim=w_dim,
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

    def test_slice_index_wraps(self):
        """Slice index should wrap modulo w_dim."""
        w_dim, depth, height, stride = 2, 3, 3, 3
        grid = np.zeros((w_dim, depth, height, stride), dtype=np.uint8)
        grid[1, 1, 1, 1] = 1  # only slice 1 has data
        
        mock_pygame, mock_screen = self._make_mocks()
        
        viewer.reset_state()
        # slice_index beyond w_dim should wrap to 0 -> empty render
        viewer.render_frame(
            mock_pygame,
            mock_screen,
            grid,
            320,
            240,
            render_mode=0,
            slice_index=2,
            width_logical=3,
            height_logical=3,
            depth_logical=3,
            w_dim=w_dim,
        )
        self.assertEqual(viewer._state.rgb_buffer.sum(), 0)
        
        # slice_index = 1 should render content
        viewer.render_frame(
            mock_pygame,
            mock_screen,
            grid,
            320,
            240,
            render_mode=0,
            slice_index=1,
            width_logical=3,
            height_logical=3,
            depth_logical=3,
            w_dim=w_dim,
        )
        self.assertTrue(viewer._state.rgb_buffer.sum() > 0)

    def test_max_intensity_contains_all_voxels(self):
        """Max projection should include voxels from different W slices."""
        w_dim, depth, height, stride = 3, 3, 3, 3
        grid = np.zeros((w_dim, depth, height, stride), dtype=np.uint8)
        grid[0, 0, 0, 0] = 1
        grid[2, 2, 2, 2] = 1
        
        mock_pygame, mock_screen = self._make_mocks()
        
        viewer.reset_state()
        viewer.render_frame(
            mock_pygame,
            mock_screen,
            grid,
            400,
            400,
            render_mode=1,  # Max-Intensity
            slice_index=0,
            width_logical=3,
            height_logical=3,
            depth_logical=3,
            w_dim=w_dim,
        )
        buf = viewer._state.rgb_buffer
        self.assertTrue(buf.sum() > 0)

    def test_tiled_depth_and_margin(self):
        """Tiled mode should expand depth and stay within margins."""
        w_dim, depth, height, stride = 4, 2, 2, 2
        grid = np.zeros((w_dim, depth, height, stride), dtype=np.uint8)
        grid[0, 0, 0, 0] = 1
        grid[3, 1, 1, 1] = 1
        
        mock_pygame, mock_screen = self._make_mocks()
        
        viewer.reset_state()
        viewer.render_frame(
            mock_pygame,
            mock_screen,
            grid,
            400,
            400,
            render_mode=2,  # Tiled
            slice_index=0,
            width_logical=2,
            height_logical=2,
            depth_logical=2,
            w_dim=w_dim,
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

    def test_mode_off_draws_nothing(self):
        """Off mode should not draw voxels."""
        w_dim, depth, height, stride = 2, 2, 2, 2
        grid = np.ones((w_dim, depth, height, stride), dtype=np.uint8)
        
        mock_pygame, mock_screen = self._make_mocks()
        
        viewer.reset_state()
        viewer.render_frame(
            mock_pygame,
            mock_screen,
            grid,
            320,
            240,
            render_mode=3,  # Off
            slice_index=0,
            width_logical=2,
            height_logical=2,
            depth_logical=2,
            w_dim=w_dim,
        )
        self.assertEqual(viewer._state.rgb_buffer.sum(), 0)


if __name__ == "__main__":
    unittest.main()


