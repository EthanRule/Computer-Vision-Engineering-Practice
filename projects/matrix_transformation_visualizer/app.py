import pygame
import numpy as np
from math_engine import MathEngine

class MatrixVisualizerApp:
    """Main application class using pygame"""
    
    def __init__(self):
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Matrix Transformation Visualizer")
        
        # Colors
        self.colors = {
            'background': (40, 40, 50),
            'panel': (60, 60, 70),
            'button': (80, 80, 90),
            'button_hover': (100, 100, 110),
            'button_active': (50, 50, 60),
            'text': (255, 255, 255),
            'grid': (80, 80, 80),
            'axis': (120, 120, 120),
            'axis_x': (220, 70, 70),    # Red for X axis
            'axis_y': (70, 220, 70),    # Green for Y axis
            'axis_z': (70, 70, 220),    # Blue for Z axis
            'original': (70, 130, 220),
            'transformed': (220, 70, 70)
        }
        
        # UI Layout
        self.panel_width = 250
        self.viewport_rect = pygame.Rect(self.panel_width, 0, self.width - self.panel_width, self.height)
        
        # Initialize components
        self.math_engine = MathEngine()
        self.ui_manager = UIManager(self.colors)
        self.renderer = Renderer(self.colors, self.viewport_rect)
        
        # View mode
        self.is_3d_mode = True
        self.view_rotation_x = np.radians(20)  # Initial view angle
        self.view_rotation_y = np.radians(30)
        
        # Camera controls
        self.camera_zoom = 50
        self.camera_offset = [0, 0]
        self.dragging = False
        self.drag_start = (0, 0)
        
        # Create UI elements
        self._create_ui()
        
        # Test points - cube for 3D, square for 2D
        self.original_points_3d = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Top face
        ])
        
        self.original_points_2d = np.array([
            [-1, -1], [1, -1], [1, 1], [-1, 1]  # Square
        ])
        
        self.clock = pygame.time.Clock()
        self.running = True
        
    def _create_ui(self):
        """Create UI elements"""
        self.ui_elements = []
        
        # Title
        title = TextLabel("Matrix Visualizer", 10, 10, 24, self.colors['text'])
        self.ui_elements.append(title)
        
        # 2D/3D Toggle
        toggle_3d = Button("3D Mode", 10, 40, 230, 30, self.colors, self.toggle_3d_mode)
        self.ui_elements.append(toggle_3d)
        self.toggle_button = toggle_3d  # Keep reference for updating text
        
        # Buttons
        buttons = [
            Button("Set Matrix", 10, 80, 230, 40, self.colors, self.on_set_matrix),
            Button("Rotate", 10, 130, 110, 40, self.colors, self.on_rotate),
            Button("Scale", 130, 130, 110, 40, self.colors, self.on_scale),
            Button("Translate", 10, 180, 110, 40, self.colors, self.on_translate),
            Button("Reset", 130, 180, 110, 40, self.colors, self.on_reset),
            Button("Undo", 10, 230, 110, 40, self.colors, self.on_undo),
            Button("Redo", 130, 230, 110, 40, self.colors, self.on_redo)
        ]
        
        # Add view control buttons for 3D mode
        view_buttons = [
            Button("View: Reset", 10, 280, 110, 30, self.colors, self.reset_view),
            Button("View: Rotate", 130, 280, 110, 30, self.colors, self.rotate_view)
        ]
        
        self.ui_elements.extend(buttons)
        self.ui_elements.extend(view_buttons)
        self.view_buttons = view_buttons  # Keep reference for showing/hiding
        
        # Matrix display
        matrix_display = MatrixDisplay(10, 320, 230, 150, self.colors, self.math_engine)
        self.ui_elements.append(matrix_display)
    
    def toggle_3d_mode(self):
        """Toggle between 2D and 3D mode"""
        self.is_3d_mode = not self.is_3d_mode
        self.toggle_button.text = "3D Mode" if self.is_3d_mode else "2D Mode"
        
        # Reset transformations when switching modes
        self.math_engine.reset_matrix()
        
        print(f"Switched to {'3D' if self.is_3d_mode else '2D'} mode")
    
    def reset_view(self):
        """Reset the 3D view angle"""
        self.view_rotation_x = np.radians(20)
        self.view_rotation_y = np.radians(30)
        print("View reset")
    
    def rotate_view(self):
        """Rotate the 3D view"""
        self.view_rotation_y += np.radians(15)
        print(f"View rotated to Y: {np.degrees(self.view_rotation_y):.1f}°")
    
    def get_current_points(self):
        """Get the current points based on mode"""
        return self.original_points_3d if self.is_3d_mode else self.original_points_2d
    
    def on_set_matrix(self):
        """Handle set matrix button"""
        print("Set Matrix clicked - would open matrix input dialog")
        
    def on_rotate(self):
        """Handle rotate button"""
        if self.is_3d_mode:
            rotation = self.math_engine.create_rotation_matrix(30)  # 30 degrees around Z
        else:
            rotation = self.math_engine.create_rotation_matrix(30)  # 2D rotation
        print("Rotation matrix:")
        print(rotation)
        self.math_engine.apply_transformation(rotation)
        
    def on_scale(self):
        """Handle scale button"""
        if self.is_3d_mode:
            scale = self.math_engine.create_scaling_matrix(1.2, 1.2, 1.2)
        else:
            scale = self.math_engine.create_scaling_matrix(1.2, 1.2)
        print("Scale matrix:")
        print(scale)
        self.math_engine.apply_transformation(scale)
        
    def on_translate(self):
        """Handle translate button"""
        if self.is_3d_mode:
            # For 3D, we'll use a hack with the 3x3 matrix (only translates X and Y)
            translation = self.math_engine.create_translation_matrix(0.5, 0.5)
        else:
            translation = self.math_engine.create_translation_matrix(0.5, 0.5)
        print("Translation matrix:")
        print(translation)
        self.math_engine.apply_transformation(translation)
        
    def on_reset(self):
        """Handle reset button"""
        self.math_engine.reset_matrix()
        
    def on_undo(self):
        """Handle undo button"""
        self.math_engine.undo()
        
    def on_redo(self):
        """Handle redo button"""
        self.math_engine.redo()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check UI elements first
                    clicked_ui = False
                    for element in self.ui_elements:
                        if hasattr(element, 'handle_click'):
                            if element.handle_click(event.pos):
                                clicked_ui = True
                                break
                    
                    # If not clicking UI, start dragging viewport
                    if not clicked_ui and self.viewport_rect.collidepoint(event.pos):
                        self.dragging = True
                        self.drag_start = event.pos
                        
                elif event.button == 4:  # Mouse wheel up
                    if self.viewport_rect.collidepoint(event.pos):
                        self.camera_zoom *= 1.1
                        
                elif event.button == 5:  # Mouse wheel down
                    if self.viewport_rect.collidepoint(event.pos):
                        self.camera_zoom *= 0.9
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
                    
            elif event.type == pygame.MOUSEMOTION:
                # Update hover states for UI elements
                for element in self.ui_elements:
                    if hasattr(element, 'update_hover'):
                        element.update_hover(event.pos)
                
                # Handle viewport dragging
                if self.dragging:
                    dx = event.pos[0] - self.drag_start[0]
                    dy = event.pos[1] - self.drag_start[1]
                    
                    if self.is_3d_mode:
                        # In 3D mode, dragging rotates the view
                        self.view_rotation_y += dx * 0.01
                        self.view_rotation_x += dy * 0.01
                        # Clamp X rotation to avoid flipping
                        self.view_rotation_x = max(-np.pi/2, min(np.pi/2, self.view_rotation_x))
                    else:
                        # In 2D mode, dragging pans the view
                        self.camera_offset[0] += dx
                        self.camera_offset[1] += dy
                    
                    self.drag_start = event.pos
    
    def update(self):
        """Update application state"""
        # Update UI elements
        for element in self.ui_elements:
            if hasattr(element, 'update'):
                element.update()
        
        # Show/hide view buttons based on mode
        for button in self.view_buttons:
            button.visible = self.is_3d_mode
    
    def render(self):
        """Render everything"""
        self.screen.fill(self.colors['background'])
        
        # Draw left panel
        panel_rect = pygame.Rect(0, 0, self.panel_width, self.height)
        pygame.draw.rect(self.screen, self.colors['panel'], panel_rect)
        
        # Draw UI elements
        for element in self.ui_elements:
            # Skip view buttons in 2D mode
            if hasattr(element, 'visible') and not element.visible:
                continue
            element.draw(self.screen)
        
        # Draw viewport
        current_points = self.get_current_points()
        if self.is_3d_mode:
            self.renderer.draw_3d_viewport(self.screen, current_points, 
                                         self.math_engine, self.camera_zoom, self.camera_offset,
                                         self.view_rotation_x, self.view_rotation_y)
        else:
            self.renderer.draw_2d_viewport(self.screen, current_points, 
                                         self.math_engine, self.camera_zoom, self.camera_offset)
        
        pygame.display.flip()
    
    def run(self):
        """Main application loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()


class UIManager:
    """Manages UI state and interactions"""
    
    def __init__(self, colors):
        self.colors = colors
        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)


class Renderer:
    """Handles 3D rendering and visualization"""
    
    def __init__(self, colors, viewport_rect):
        self.colors = colors
        self.viewport_rect = viewport_rect
        self.center_x = viewport_rect.width // 2
        self.center_y = viewport_rect.height // 2
    
    def apply_view_rotation(self, points_3d, rot_x, rot_y):
        """Apply view rotation to 3D points"""
        # Rotation around X axis
        cos_x, sin_x = np.cos(rot_x), np.sin(rot_x)
        rot_matrix_x = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        
        # Rotation around Y axis
        cos_y, sin_y = np.cos(rot_y), np.sin(rot_y)
        rot_matrix_y = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        
        # Apply rotations
        view_matrix = rot_matrix_y @ rot_matrix_x
        return np.array([view_matrix @ point for point in points_3d])
    
    def project_3d_to_2d(self, points_3d, zoom, offset):
        """Simple orthographic projection"""
        projected = []
        for point in points_3d:
            x = point[0] * zoom + self.center_x + offset[0]
            y = -point[1] * zoom + self.center_y + offset[1]  # Flip Y for screen coords
            projected.append((x + self.viewport_rect.x, y + self.viewport_rect.y))
        return projected
    
    def project_2d_to_screen(self, points_2d, zoom, offset):
        """Project 2D points to screen coordinates"""
        projected = []
        for point in points_2d:
            x = point[0] * zoom + self.center_x + offset[0]
            y = -point[1] * zoom + self.center_y + offset[1]  # Flip Y for screen coords
            projected.append((x + self.viewport_rect.x, y + self.viewport_rect.y))
        return projected
    
    def draw_grid(self, surface, zoom, offset):
        """Draw grid in viewport"""
        grid_size = 50
        
        # Vertical lines
        for i in range(-10, 11):
            x = i * grid_size * zoom + self.center_x + offset[0] + self.viewport_rect.x
            if self.viewport_rect.left <= x <= self.viewport_rect.right:
                pygame.draw.line(surface, self.colors['grid'], 
                               (x, self.viewport_rect.top), 
                               (x, self.viewport_rect.bottom))
        
        # Horizontal lines
        for i in range(-10, 11):
            y = i * grid_size * zoom + self.center_y + offset[1] + self.viewport_rect.y
            if self.viewport_rect.top <= y <= self.viewport_rect.bottom:
                pygame.draw.line(surface, self.colors['grid'], 
                               (self.viewport_rect.left, y), 
                               (self.viewport_rect.right, y))
    
    def draw_3d_axes(self, surface, zoom, offset, rot_x, rot_y):
        """Draw 3D coordinate axes"""
        # Define axis vectors
        axis_length = 2.0
        axes_3d = np.array([
            [0, 0, 0], [axis_length, 0, 0],  # X axis
            [0, 0, 0], [0, axis_length, 0],  # Y axis
            [0, 0, 0], [0, 0, axis_length]   # Z axis
        ])
        
        # Apply view rotation
        axes_rotated = self.apply_view_rotation(axes_3d, rot_x, rot_y)
        axes_2d = self.project_3d_to_2d(axes_rotated, zoom, offset)
        
        # Draw axes
        colors = [self.colors['axis_x'], self.colors['axis_y'], self.colors['axis_z']]
        labels = ['X', 'Y', 'Z']
        
        font = pygame.font.Font(None, 24)
        
        for i in range(3):
            start_idx = i * 2
            end_idx = i * 2 + 1
            
            start_point = axes_2d[start_idx]
            end_point = axes_2d[end_idx]
            
            # Draw axis line
            pygame.draw.line(surface, colors[i], start_point, end_point, 3)
            
            # Draw axis label
            label = font.render(labels[i], True, colors[i])
            label_pos = (end_point[0] + 10, end_point[1] - 10)
            surface.blit(label, label_pos)
    
    def draw_2d_axes(self, surface, zoom, offset):
        """Draw 2D coordinate axes"""
        axis_x = self.center_x + offset[0] + self.viewport_rect.x
        axis_y = self.center_y + offset[1] + self.viewport_rect.y
        
        # X axis
        if self.viewport_rect.left <= axis_x <= self.viewport_rect.right:
            pygame.draw.line(surface, self.colors['axis_x'], 
                           (axis_x, self.viewport_rect.top), 
                           (axis_x, self.viewport_rect.bottom), 2)
        
        # Y axis
        if self.viewport_rect.top <= axis_y <= self.viewport_rect.bottom:
            pygame.draw.line(surface, self.colors['axis_y'], 
                           (self.viewport_rect.left, axis_y), 
                           (self.viewport_rect.right, axis_y), 2)
        
        # Axis labels
        font = pygame.font.Font(None, 24)
        x_label = font.render("X", True, self.colors['axis_x'])
        y_label = font.render("Y", True, self.colors['axis_y'])
        
        surface.blit(x_label, (self.viewport_rect.right - 30, axis_y + 10))
        surface.blit(y_label, (axis_x + 10, self.viewport_rect.top + 10))
    
    def draw_3d_viewport(self, surface, original_points, math_engine, zoom, offset, rot_x, rot_y):
        """Draw the 3D viewport"""
        # Draw grid
        self.draw_grid(surface, zoom / 50, offset)
        
        # Draw 3D axes
        self.draw_3d_axes(surface, zoom, offset, rot_x, rot_y)
        
        # Transform points
        transformed_points = math_engine.transform_points(original_points)
        
        # Apply view rotation
        original_rotated = self.apply_view_rotation(original_points, rot_x, rot_y)
        transformed_rotated = self.apply_view_rotation(transformed_points, rot_x, rot_y)
        
        # Project to 2D
        original_2d = self.project_3d_to_2d(original_rotated, zoom, offset)
        transformed_2d = self.project_3d_to_2d(transformed_rotated, zoom, offset)
        
        # Draw cube wireframe
        self._draw_cube_wireframe(surface, original_2d, self.colors['original'])
        self._draw_cube_wireframe(surface, transformed_2d, self.colors['transformed'])
        
        # Draw legend
        font = pygame.font.Font(None, 24)
        orig_text = font.render("Original", True, self.colors['original'])
        trans_text = font.render("Transformed", True, self.colors['transformed'])
        mode_text = font.render("3D Mode - Drag to rotate view", True, self.colors['text'])
        
        surface.blit(orig_text, (self.viewport_rect.x + 10, self.viewport_rect.y + 10))
        surface.blit(trans_text, (self.viewport_rect.x + 10, self.viewport_rect.y + 35))
        surface.blit(mode_text, (self.viewport_rect.x + 10, self.viewport_rect.bottom - 30))
    
    def draw_2d_viewport(self, surface, original_points, math_engine, zoom, offset):
        """Draw the 2D viewport"""
        # Draw grid
        self.draw_grid(surface, zoom / 50, offset)
        
        # Draw 2D axes
        self.draw_2d_axes(surface, zoom, offset)
        
        # Transform points
        transformed_points = math_engine.transform_points(original_points)
        
        # Project to screen
        original_2d = self.project_2d_to_screen(original_points, zoom, offset)
        transformed_2d = self.project_2d_to_screen(transformed_points, zoom, offset)
        
        # Draw square
        self._draw_2d_shape(surface, original_2d, self.colors['original'])
        self._draw_2d_shape(surface, transformed_2d, self.colors['transformed'])
        
        # Draw legend
        font = pygame.font.Font(None, 24)
        orig_text = font.render("Original", True, self.colors['original'])
        trans_text = font.render("Transformed", True, self.colors['transformed'])
        mode_text = font.render("2D Mode - Drag to pan view", True, self.colors['text'])
        
        surface.blit(orig_text, (self.viewport_rect.x + 10, self.viewport_rect.y + 10))
        surface.blit(trans_text, (self.viewport_rect.x + 10, self.viewport_rect.y + 35))
        surface.blit(mode_text, (self.viewport_rect.x + 10, self.viewport_rect.bottom - 30))
    
    def _draw_cube_wireframe(self, surface, points_2d, color):
        """Draw a cube wireframe"""
        if len(points_2d) < 8:
            return
            
        # Define cube edges (indices into points array)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        for edge in edges:
            start_point = points_2d[edge[0]]
            end_point = points_2d[edge[1]]
            
            # Only draw if both points are within reasonable bounds
            if (-100 < start_point[0] < self.viewport_rect.width + 100 and
                -100 < start_point[1] < self.viewport_rect.height + 100 and
                -100 < end_point[0] < self.viewport_rect.width + 100 and
                -100 < end_point[1] < self.viewport_rect.height + 100):
                
                pygame.draw.line(surface, color, start_point, end_point, 2)
    
    def _draw_2d_shape(self, surface, points_2d, color):
        """Draw a 2D shape (square)"""
        if len(points_2d) < 4:
            return
        
        # Draw square edges
        for i in range(len(points_2d)):
            start_point = points_2d[i]
            end_point = points_2d[(i + 1) % len(points_2d)]
            
            pygame.draw.line(surface, color, start_point, end_point, 2)


class Button:
    """A clickable button UI element"""
    
    def __init__(self, text, x, y, width, height, colors, callback=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.colors = colors
        self.callback = callback
        self.font = pygame.font.Font(None, 24)
        
        self.is_hovered = False
        self.is_pressed = False
        self.visible = True  # Add visibility control
    
    def update_hover(self, mouse_pos):
        """Update hover state"""
        if self.visible:
            self.is_hovered = self.rect.collidepoint(mouse_pos)
    
    def handle_click(self, mouse_pos):
        """Handle click event"""
        if self.visible and self.rect.collidepoint(mouse_pos):
            if self.callback:
                self.callback()
            return True
        return False
    
    def draw(self, surface):
        """Draw the button"""
        if not self.visible:
            return
            
        # Choose color based on state
        if self.is_pressed:
            color = self.colors['button_active']
        elif self.is_hovered:
            color = self.colors['button_hover']
        else:
            color = self.colors['button']
        
        # Draw button
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, self.colors['text'], self.rect, 2)
        
        # Draw text
        text_surface = self.font.render(self.text, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class TextLabel:
    """A text label UI element"""
    
    def __init__(self, text, x, y, font_size, color):
        self.text = text
        self.pos = (x, y)
        self.font = pygame.font.Font(None, font_size)
        self.color = color
    
    def draw(self, surface):
        """Draw the text label"""
        text_surface = self.font.render(self.text, True, self.color)
        surface.blit(text_surface, self.pos)


class MatrixDisplay:
    """Displays the current transformation matrix"""
    
    def __init__(self, x, y, width, height, colors, math_engine):
        self.rect = pygame.Rect(x, y, width, height)
        self.colors = colors
        self.math_engine = math_engine
        self.font = pygame.font.Font(None, 18)
    
    def draw(self, surface):
        """Draw the matrix display"""
        # Draw background
        pygame.draw.rect(surface, self.colors['button'], self.rect)
        pygame.draw.rect(surface, self.colors['text'], self.rect, 1)
        
        # Get current matrix
        matrix = self.math_engine.get_matrix()
        
        # Draw title
        title_surface = self.font.render("Current Matrix:", True, self.colors['text'])
        surface.blit(title_surface, (self.rect.x + 5, self.rect.y + 5))
        
        # Draw matrix values
        y_offset = 25
        for i in range(min(3, matrix.shape[0])):
            row_text = "["
            for j in range(min(3, matrix.shape[1])):
                value = matrix[i, j]
                row_text += f"{value:7.3f}"
                if j < min(2, matrix.shape[1] - 1):
                    row_text += ", "
            row_text += "]"
            
            text_surface = self.font.render(row_text, True, self.colors['text'])
            surface.blit(text_surface, (self.rect.x + 5, self.rect.y + y_offset))
            y_offset += 20


if __name__ == "__main__":
    app = MatrixVisualizerApp()
    app.run()
