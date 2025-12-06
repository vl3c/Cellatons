"""Tests for the renderer viewer module.

Run: python row/renderer/test_viewer.py
Or:  pytest row/renderer/test_viewer.py -v
"""

import numpy as np
import unittest
from unittest.mock import Mock, MagicMock, patch

# Import the module under test
import viewer


class TestCreateGridView(unittest.TestCase):
    """Tests for create_grid_view function."""
    
    def test_creates_correct_shape(self):
        """Grid view should have shape (height, stride)."""
        # Create a test buffer
        height, stride = 100, 128
        test_data = np.zeros(height * stride, dtype=np.uint8)
        test_data[0] = 1  # Mark first cell
        test_data[stride // 2] = 1  # Mark middle of first row
        
        # Get pointer as integer
        ptr = test_data.ctypes.data
        
        # Create view
        view = viewer.create_grid_view(ptr, stride, height, 100)
        
        self.assertEqual(view.shape, (height, stride))
    
    def test_zero_copy_view(self):
        """Changes to view should reflect in original data."""
        height, stride = 10, 16
        test_data = np.zeros(height * stride, dtype=np.uint8)
        ptr = test_data.ctypes.data
        
        view = viewer.create_grid_view(ptr, stride, height, 10)
        
        # Modify through view
        view[5, 8] = 42
        
        # Check original data changed
        self.assertEqual(test_data[5 * stride + 8], 42)
    
    def test_handles_large_grid(self):
        """Should handle large grid dimensions."""
        height, stride = 1000, 1024
        test_data = np.zeros(height * stride, dtype=np.uint8)
        ptr = test_data.ctypes.data
        
        view = viewer.create_grid_view(ptr, stride, height, 1000)
        
        self.assertEqual(view.shape, (height, stride))


class TestRenderState(unittest.TestCase):
    """Tests for RenderState class."""
    
    def setUp(self):
        viewer.reset_state()
    
    def test_initial_state_is_none(self):
        """Fresh state should have None buffers."""
        state = viewer.RenderState()
        self.assertIsNone(state.display_buffer)
        self.assertIsNone(state.rgb_buffer)
        self.assertIsNone(state.surface)
    
    def test_ensure_initialized_creates_buffers(self):
        """ensure_initialized should create correctly sized buffers."""
        state = viewer.RenderState()
        
        # Mock pygame
        mock_pygame = Mock()
        mock_pygame.Surface.return_value = Mock()
        mock_pygame.font.init = Mock()
        mock_pygame.font.SysFont.return_value = Mock()
        
        state.ensure_initialized(mock_pygame, 800, 600)
        
        self.assertEqual(state.display_buffer.shape, (600, 800))
        self.assertEqual(state.rgb_buffer.shape, (800, 600, 3))
        self.assertEqual(state.display_width, 800)
        self.assertEqual(state.display_height, 600)
    
    def test_ensure_initialized_resizes_on_change(self):
        """Buffers should resize when dimensions change."""
        state = viewer.RenderState()
        mock_pygame = Mock()
        mock_pygame.Surface.return_value = Mock()
        mock_pygame.font.init = Mock()
        mock_pygame.font.SysFont.return_value = Mock()
        
        # First init
        state.ensure_initialized(mock_pygame, 800, 600)
        self.assertEqual(state.display_width, 800)
        
        # Resize
        state.ensure_initialized(mock_pygame, 1920, 1080)
        self.assertEqual(state.display_buffer.shape, (1080, 1920))
        self.assertEqual(state.display_width, 1920)


class TestRenderGrid(unittest.TestCase):
    """Tests for _render_grid function."""
    
    def setUp(self):
        viewer.reset_state()
        # Initialize state with mock pygame
        mock_pygame = Mock()
        mock_pygame.Surface.return_value = Mock()
        mock_pygame.font.init = Mock()
        mock_pygame.font.SysFont.return_value = Mock()
        viewer._state.ensure_initialized(mock_pygame, 100, 50)
    
    def test_extracts_correct_region(self):
        """Should extract visible region from grid."""
        # Create test grid with pattern
        grid = np.zeros((200, 128), dtype=np.uint8)
        grid[10:20, 14:24] = 1  # Block of 1s
        
        viewer._render_grid(grid, start_row=10, display_width=100, 
                           display_height=50, view_left=14)
        
        # Check that the block appears at (0:10, 0:10) in display buffer
        # After scaling by 255
        self.assertEqual(viewer._state.display_buffer[0, 0], 255)
        self.assertEqual(viewer._state.display_buffer[9, 9], 255)
        self.assertEqual(viewer._state.display_buffer[10, 0], 0)  # Past the block
    
    def test_handles_end_of_grid(self):
        """Should handle scroll position near end of grid."""
        grid = np.ones((100, 128), dtype=np.uint8)
        
        # Start at row 80, display height 50 -> only 20 rows available
        viewer._render_grid(grid, start_row=80, display_width=100,
                           display_height=50, view_left=0)
        
        # First 20 rows should have data, rest should be 0
        self.assertEqual(viewer._state.display_buffer[19, 0], 255)
        self.assertEqual(viewer._state.display_buffer[20, 0], 0)


class TestRenderWindow(unittest.TestCase):
    """Integration tests for render_window function."""
    
    def setUp(self):
        viewer.reset_state()
    
    def test_renders_without_error(self):
        """Full render should complete without raising."""
        # Create test grid
        grid = np.zeros((100, 128), dtype=np.uint8)
        grid[25, 64] = 1
        
        # Mock pygame objects
        mock_pygame = Mock()
        mock_pygame.Surface.return_value = Mock()
        mock_pygame.font.init = Mock()
        mock_font = Mock()
        mock_font.render.return_value = Mock(get_rect=Mock(return_value=Mock()))
        mock_pygame.font.SysFont.return_value = mock_font
        mock_pygame.surfarray.blit_array = Mock()
        
        mock_screen = Mock()
        
        # Should not raise
        viewer.render_window(
            mock_pygame, mock_screen, grid,
            start_row=0, display_width=100, display_height=50,
            view_left=14, progress=50, fps=60
        )
        
        # Verify blit was called
        mock_screen.blit.assert_called()
    
    def test_status_text_format(self):
        """Status text should show progress and FPS."""
        grid = np.zeros((100, 128), dtype=np.uint8)
        
        mock_pygame = Mock()
        mock_pygame.Surface.return_value = Mock()
        mock_pygame.font.init = Mock()
        mock_font = Mock()
        mock_font.render.return_value = Mock(get_rect=Mock(return_value=Mock()))
        mock_pygame.font.SysFont.return_value = mock_font
        mock_pygame.surfarray.blit_array = Mock()
        mock_screen = Mock()
        
        viewer.render_window(
            mock_pygame, mock_screen, grid,
            start_row=0, display_width=100, display_height=50,
            view_left=0, progress=75, fps=120
        )
        
        # Check font.render was called with correct text
        mock_font.render.assert_called_once()
        call_args = mock_font.render.call_args[0]
        self.assertIn("75%", call_args[0])
        self.assertIn("120 FPS", call_args[0])


class TestResetState(unittest.TestCase):
    """Tests for reset_state function."""
    
    def test_reset_clears_buffers(self):
        """reset_state should create fresh state."""
        # Initialize some state
        mock_pygame = Mock()
        mock_pygame.Surface.return_value = Mock()
        mock_pygame.font.init = Mock()
        mock_pygame.font.SysFont.return_value = Mock()
        viewer._state.ensure_initialized(mock_pygame, 800, 600)
        
        self.assertIsNotNone(viewer._state.display_buffer)
        
        # Reset
        viewer.reset_state()
        
        self.assertIsNone(viewer._state.display_buffer)


if __name__ == "__main__":
    unittest.main()

