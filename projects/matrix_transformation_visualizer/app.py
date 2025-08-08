import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
from tkinter import messagebox, scrolledtext
import tkinter as tk
from DataEngine import DataEngine

# Configure the appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MatrixTransformationApp:
    def __init__(self):
        # Initialize the data engine
        self.engine = DataEngine()
        
        # Clean, professional color scheme with good contrast
        self.colors = {
            'primary': '#2563eb',      # Clean blue
            'primary_hover': '#1d4ed8', # Darker blue for hover
            'secondary': '#64748b',     # Slate gray
            'background': '#f8fafc',    # Light gray background
            'surface': '#ffffff',       # White surface
            'text_primary': '#1e293b',  # Dark gray text
            'text_secondary': '#64748b', # Medium gray text
            'border': '#e2e8f0'        # Light border
        }
        
        # Setup GUI
        self.setup_gui()
        self.update_display()
    
    def setup_gui(self):
        """Initialize the GUI components with clean, professional styling"""
        self.root = ctk.CTk()
        self.root.title("Matrix Transformation Visualizer")
        self.root.geometry("1400x900")
        self.root.configure(fg_color=("#f8fafc", "#1e293b"))
        
        # Configure window properties
        self.root.minsize(1200, 800)
        
        # Create main container
        main_container = ctk.CTkFrame(
            self.root, 
            fg_color="transparent",
            corner_radius=0
        )
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Configure grid weights
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)
        
        # Left panel for controls
        self.setup_left_panel(main_container)
        
        # Right panel for visualization
        self.setup_right_panel(main_container)
    
    def setup_left_panel(self, parent):
        """Setup the left control panel with clean styling"""
        left_panel = ctk.CTkScrollableFrame(
            parent,
            width=420,
            fg_color=("#ffffff", "#334155"),
            corner_radius=12,
            border_width=1,
            border_color=("#e2e8f0", "#475569")
        )
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        
        # Header
        header_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Matrix Transformation Lab",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=("#1e293b", "#f1f5f9")
        )
        title_label.pack(pady=15)
        
        # Matrix Creation Section
        self.setup_matrix_creation(left_panel)
        
        # Matrix Input Section
        self.setup_matrix_input(left_panel)
        
        # Transformation Buttons
        self.setup_transformation_buttons(left_panel)
        
        # Matrix Operations
        self.setup_matrix_operations(left_panel)
        
        # Advanced Operations
        self.setup_advanced_operations(left_panel)
        
        # History and Undo/Redo
        self.setup_history_controls(left_panel)
        
        # Matrix Info Display
        self.setup_info_display(left_panel)
    
    def create_section_frame(self, parent, title):
        """Create a clean section frame with minimal styling"""
        section = ctk.CTkFrame(
            parent,
            fg_color=("#f8fafc", "#475569"),
            corner_radius=8,
            border_width=1,
            border_color=("#e2e8f0", "#64748b")
        )
        section.pack(fill="x", pady=(0, 15), padx=15)
        
        # Section header
        title_label = ctk.CTkLabel(
            section,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1e293b", "#f1f5f9")
        )
        title_label.pack(pady=(15, 10))
        
        return section

    def create_button(self, parent, text, command, width=120, height=32, style="primary"):
        """Create a clean, professional button"""
        if style == "primary":
            fg_color = "#2563eb"
            hover_color = "#1d4ed8"
        else:  # secondary
            fg_color = "#64748b"
            hover_color = "#475569"
            
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width,
            height=height,
            corner_radius=6,
            fg_color=fg_color,
            hover_color=hover_color,
            text_color="#ffffff",
            font=ctk.CTkFont(size=13, weight="normal"),
            border_width=0
        )
    def setup_matrix_creation(self, parent):
        """Setup matrix creation controls for arbitrary dimensions"""
        creation_frame = self.create_section_frame(parent, "Create Matrix")
        
        # Dimensions input
        dim_frame = ctk.CTkFrame(creation_frame, fg_color="transparent")
        dim_frame.pack(padx=15, pady=(0, 10), fill="x")
        
        ctk.CTkLabel(
            dim_frame, 
            text="Dimensions:", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#374151", "#9ca3af")
        ).pack(anchor="w")
        
        # Row and column inputs
        size_container = ctk.CTkFrame(dim_frame, fg_color="transparent")
        size_container.pack(fill="x", pady=(5, 0))
        
        ctk.CTkLabel(size_container, text="Rows:", text_color=("#6b7280", "#9ca3af")).pack(side="left")
        self.rows_entry = ctk.CTkEntry(size_container, width=50, height=24)
        self.rows_entry.pack(side="left", padx=(5, 15))
        self.rows_entry.insert(0, "2")
        
        ctk.CTkLabel(size_container, text="Cols:", text_color=("#6b7280", "#9ca3af")).pack(side="left")
        self.cols_entry = ctk.CTkEntry(size_container, width=50, height=24)
        self.cols_entry.pack(side="left", padx=(5, 0))
        self.cols_entry.insert(0, "2")
        
        # Matrix creation buttons
        button_container = ctk.CTkFrame(creation_frame, fg_color="transparent")
        button_container.pack(padx=15, pady=(5, 15), fill="x")
        
        # Row 1
        row1 = ctk.CTkFrame(button_container, fg_color="transparent")
        row1.pack(fill="x", pady=2)
        
        self.create_button(row1, "Identity", self.create_identity, width=85, height=28).pack(side="left", padx=(0, 5))
        self.create_button(row1, "Random", self.create_random, width=85, height=28).pack(side="left", padx=5)
        self.create_button(row1, "Zeros", self.create_zeros, width=85, height=28).pack(side="left", padx=(5, 0))
        
        # Row 2
        row2 = ctk.CTkFrame(button_container, fg_color="transparent")
        row2.pack(fill="x", pady=(5, 0))
        
        self.create_button(row2, "Ones", self.create_ones, width=85, height=28, style="secondary").pack(side="left", padx=(0, 5))
        self.create_button(row2, "2x2 Default", self.create_default_2x2, width=85, height=28, style="secondary").pack(side="left", padx=5)
        self.create_button(row2, "6x2 Example", self.create_6x2_example, width=85, height=28, style="secondary").pack(side="left", padx=(5, 0))

    def setup_matrix_input(self, parent):
        """Setup matrix input controls with clean styling"""
        matrix_frame = self.create_section_frame(parent, "Transformation Matrix")
        
        # Matrix entries container
        entries_container = ctk.CTkFrame(matrix_frame, fg_color="transparent")
        entries_container.pack(pady=(0, 15))
        
        # Matrix bracket visual
        bracket_frame = ctk.CTkFrame(entries_container, fg_color="transparent")
        bracket_frame.pack(pady=10)
        
        # Left bracket
        ctk.CTkLabel(
            bracket_frame, 
            text="[", 
            font=ctk.CTkFont(size=32), 
            text_color=("#64748b", "#94a3b8")
        ).grid(row=0, column=0, rowspan=2, padx=(0, 8))
        
        # Matrix entries with clean styling
        entry_style = {
            "width": 80,
            "height": 32,
            "corner_radius": 6,
            "border_width": 1,
            "border_color": ("#d1d5db", "#64748b"),
            "fg_color": ("#ffffff", "#334155"),
            "text_color": ("#1e293b", "#f1f5f9"),
            "font": ctk.CTkFont(size=14)
        }
        
        self.a11 = ctk.CTkEntry(bracket_frame, **entry_style)
        self.a12 = ctk.CTkEntry(bracket_frame, **entry_style)
        self.a21 = ctk.CTkEntry(bracket_frame, **entry_style)
        self.a22 = ctk.CTkEntry(bracket_frame, **entry_style)
        
        # Set default values (identity matrix)
        self.a11.insert(0, "1.0")
        self.a12.insert(0, "0.0")
        self.a21.insert(0, "0.0")
        self.a22.insert(0, "1.0")
        
        # Grid layout
        self.a11.grid(row=0, column=1, padx=4, pady=4)
        self.a12.grid(row=0, column=2, padx=4, pady=4)
        self.a21.grid(row=1, column=1, padx=4, pady=4)
        self.a22.grid(row=1, column=2, padx=4, pady=4)
        
        # Right bracket
        ctk.CTkLabel(
            bracket_frame, 
            text="]", 
            font=ctk.CTkFont(size=32), 
            text_color=("#64748b", "#94a3b8")
        ).grid(row=0, column=3, rowspan=2, padx=(8, 0))
        
        # Apply button
        apply_btn = self.create_button(
            matrix_frame, 
            "Apply Matrix", 
            self.apply_custom_matrix, 
            width=140,
            height=36
        )
        apply_btn.pack(pady=(0, 15))
    
    def setup_transformation_buttons(self, parent):
        """Setup preset transformation buttons with create/apply modes"""
        trans_frame = self.create_section_frame(parent, "Transformations")
        
        # Mode selection
        mode_frame = ctk.CTkFrame(trans_frame, fg_color="transparent")
        mode_frame.pack(padx=15, pady=(0, 10), fill="x")
        
        ctk.CTkLabel(
            mode_frame, 
            text="Mode:", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#374151", "#9ca3af")
        ).pack(side="left")
        
        self.transform_mode = ctk.StringVar(value="create")
        
        create_radio = ctk.CTkRadioButton(
            mode_frame, 
            text="Create Matrix", 
            variable=self.transform_mode, 
            value="create",
            text_color=("#374151", "#9ca3af")
        )
        create_radio.pack(side="left", padx=(10, 15))
        
        apply_radio = ctk.CTkRadioButton(
            mode_frame, 
            text="Apply to Current", 
            variable=self.transform_mode, 
            value="apply",
            text_color=("#374151", "#9ca3af")
        )
        apply_radio.pack(side="left")
        
        # Rotation section
        rot_frame = ctk.CTkFrame(trans_frame, fg_color="transparent")
        rot_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(
            rot_frame, 
            text="Rotation (degrees):", 
            font=ctk.CTkFont(size=13),
            text_color=("#1e293b", "#f1f5f9")
        ).pack(side="left")
        
        self.rotation_entry = ctk.CTkEntry(
            rot_frame, 
            width=80, 
            height=28,
            corner_radius=4,
            font=ctk.CTkFont(size=12),
            fg_color=("#ffffff", "#334155"),
            text_color=("#1e293b", "#f1f5f9"),
            border_color=("#d1d5db", "#64748b")
        )
        self.rotation_entry.pack(side="left", padx=(8, 8))
        self.rotation_entry.insert(0, "45")
        
        self.create_button(rot_frame, "Rotate", self.apply_rotation, width=80, height=28).pack(side="left")
        
        # Scaling section
        scale_frame = ctk.CTkFrame(trans_frame, fg_color="transparent")
        scale_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(
            scale_frame, 
            text="Scale X:", 
            font=ctk.CTkFont(size=13),
            text_color=("#1e293b", "#f1f5f9")
        ).pack(side="left")
        
        self.scale_x_entry = ctk.CTkEntry(
            scale_frame, 
            width=60, 
            height=28,
            corner_radius=4,
            font=ctk.CTkFont(size=12),
            fg_color=("#ffffff", "#334155"),
            text_color=("#1e293b", "#f1f5f9"),
            border_color=("#d1d5db", "#64748b")
        )
        self.scale_x_entry.pack(side="left", padx=(4, 8))
        self.scale_x_entry.insert(0, "1.5")
        
        ctk.CTkLabel(
            scale_frame, 
            text="Y:", 
            font=ctk.CTkFont(size=13),
            text_color=("#1e293b", "#f1f5f9")
        ).pack(side="left")
        
        self.scale_y_entry = ctk.CTkEntry(
            scale_frame, 
            width=60, 
            height=28,
            corner_radius=4,
            font=ctk.CTkFont(size=12),
            fg_color=("#ffffff", "#334155"),
            text_color=("#1e293b", "#f1f5f9"),
            border_color=("#d1d5db", "#64748b")
        )
        self.scale_y_entry.pack(side="left", padx=(4, 8))
        self.scale_y_entry.insert(0, "1.5")
        
        self.create_button(scale_frame, "Scale", self.apply_scaling, width=80, height=28).pack(side="left")
        
        # Reflection buttons
        ref_frame = ctk.CTkFrame(trans_frame, fg_color="transparent")
        ref_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(
            ref_frame, 
            text="Reflect:", 
            font=ctk.CTkFont(size=13),
            text_color=("#1e293b", "#f1f5f9")
        ).pack(side="left")
        
        self.create_button(ref_frame, "X-axis", lambda: self.apply_reflection("x"), width=70, height=28, style="secondary").pack(side="left", padx=(8, 4))
        self.create_button(ref_frame, "Y-axis", lambda: self.apply_reflection("y"), width=70, height=28, style="secondary").pack(side="left", padx=2)
        self.create_button(ref_frame, "XY-line", lambda: self.apply_reflection("xy"), width=70, height=28, style="secondary").pack(side="left", padx=(2, 0))
        
        # Shear section
        shear_frame = ctk.CTkFrame(trans_frame, fg_color="transparent")
        shear_frame.pack(fill="x", padx=15, pady=(5, 15))
        
        ctk.CTkLabel(
            shear_frame, 
            text="Shear X:", 
            font=ctk.CTkFont(size=13),
            text_color=("#1e293b", "#f1f5f9")
        ).pack(side="left")
        
        self.shear_x_entry = ctk.CTkEntry(
            shear_frame, 
            width=60, 
            height=28,
            corner_radius=4,
            font=ctk.CTkFont(size=12),
            fg_color=("#ffffff", "#334155"),
            text_color=("#1e293b", "#f1f5f9"),
            border_color=("#d1d5db", "#64748b")
        )
        self.shear_x_entry.pack(side="left", padx=(4, 8))
        self.shear_x_entry.insert(0, "0.5")
        
        ctk.CTkLabel(
            shear_frame, 
            text="Y:", 
            font=ctk.CTkFont(size=13),
            text_color=("#1e293b", "#f1f5f9")
        ).pack(side="left")
        
        self.shear_y_entry = ctk.CTkEntry(
            shear_frame, 
            width=60, 
            height=28,
            corner_radius=4,
            font=ctk.CTkFont(size=12),
            fg_color=("#ffffff", "#334155"),
            text_color=("#1e293b", "#f1f5f9"),
            border_color=("#d1d5db", "#64748b")
        )
        self.shear_y_entry.pack(side="left", padx=(4, 8))
        self.shear_y_entry.insert(0, "0.5")
        
        self.create_button(shear_frame, "Shear", self.apply_shear, width=80, height=28).pack(side="left")

    def setup_matrix_operations(self, parent):
        """Setup basic matrix operations with clean layout"""
        ops_frame = self.create_section_frame(parent, "Matrix Operations")
        
        # Button grid container
        button_container = ctk.CTkFrame(ops_frame, fg_color="transparent")
        button_container.pack(padx=15, pady=(0, 15))
        
        # Row 1
        row1 = ctk.CTkFrame(button_container, fg_color="transparent")
        row1.pack(fill="x", pady=2)
        
        self.create_button(row1, "Transpose", self.transpose_matrix, width=100, height=32).pack(side="left", padx=(0, 8))
        self.create_button(row1, "Inverse", self.inverse_matrix, width=100, height=32).pack(side="left", padx=4)
        self.create_button(row1, "Determinant", self.calculate_determinant, width=100, height=32).pack(side="left", padx=(8, 0))
        
        # Row 2
        row2 = ctk.CTkFrame(button_container, fg_color="transparent")
        row2.pack(fill="x", pady=2)
        
        self.create_button(row2, "Reset", self.reset_matrix, width=100, height=32, style="secondary").pack(side="left", padx=(0, 8))
        self.create_button(row2, "Random", self.generate_random, width=100, height=32, style="secondary").pack(side="left", padx=4)
        self.create_button(row2, "Identity", self.set_identity, width=100, height=32, style="secondary").pack(side="left", padx=(8, 0))

    def setup_advanced_operations(self, parent):
        """Setup advanced matrix operations with clean layout"""
        advanced_frame = self.create_section_frame(parent, "Advanced Operations")
        
        # Button grid for advanced operations
        button_container = ctk.CTkFrame(advanced_frame, fg_color="transparent")
        button_container.pack(padx=15, pady=(0, 15))
        
        # Row 1
        row1 = ctk.CTkFrame(button_container, fg_color="transparent")
        row1.pack(fill="x", pady=2)
        
        self.create_button(row1, "Eigenvalues", self.show_eigenvalues, width=130, height=32).pack(side="left", padx=(0, 8))
        self.create_button(row1, "SVD Analysis", self.show_svd, width=130, height=32).pack(side="left", padx=(8, 0))
        
        # Row 2
        row2 = ctk.CTkFrame(button_container, fg_color="transparent")
        row2.pack(fill="x", pady=2)
        
        self.create_button(row2, "Matrix Info", self.show_matrix_info, width=130, height=32, style="secondary").pack(side="left", padx=(0, 8))
        self.create_button(row2, "Dot Product", self.show_dot_product, width=130, height=32, style="secondary").pack(side="left", padx=(8, 0))

    def setup_history_controls(self, parent):
        """Setup undo/redo and history controls with clean styling"""
        history_frame = self.create_section_frame(parent, "History Control")
        
        # Main control buttons
        button_container = ctk.CTkFrame(history_frame, fg_color="transparent")
        button_container.pack(padx=15, pady=(0, 15))
        
        # Button row
        button_row = ctk.CTkFrame(button_container, fg_color="transparent")
        button_row.pack()
        
        self.undo_btn = self.create_button(button_row, "← Undo", self.undo, width=90, height=32, style="secondary")
        self.undo_btn.pack(side="left", padx=(0, 8))
        
        self.redo_btn = self.create_button(button_row, "Redo →", self.redo, width=90, height=32, style="secondary")
        self.redo_btn.pack(side="left", padx=4)
        
        self.create_button(button_row, "Clear", self.clear_history, width=90, height=32, style="secondary").pack(side="left", padx=(8, 0))

    def setup_info_display(self, parent):
        """Setup information display area with clean styling"""
        info_frame = self.create_section_frame(parent, "Matrix Information")
        
        # Text display container
        text_container = ctk.CTkFrame(info_frame, fg_color="transparent")
        text_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Clean text display
        self.info_text = ctk.CTkTextbox(
            text_container,
            height=150,
            corner_radius=6,
            fg_color=("#ffffff", "#334155"),
            text_color=("#1e293b", "#f1f5f9"),
            border_width=1,
            border_color=("#d1d5db", "#64748b"),
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.info_text.pack(fill="both", expand=True)
        
        # Set initial text
        self.info_text.insert("0.0", "Matrix information will appear here...")
        self.info_text.configure(state="disabled")
    
    def setup_right_panel(self, parent):
        """Setup the visualization panel with clean styling"""
        right_panel = ctk.CTkFrame(
            parent,
            fg_color=("#ffffff", "#334155"),
            corner_radius=12,
            border_width=1,
            border_color=("#e2e8f0", "#64748b")
        )
        right_panel.grid(row=0, column=1, sticky="nsew")
        
        # Header
        header_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Transformation Visualization",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1e293b", "#f1f5f9")
        )
        title_label.pack()
        
        # Create matplotlib figure with proper styling
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.patch.set_facecolor('white')
        
        # Configure the plot
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.grid(True, alpha=0.3, color='#d1d5db')
        self.ax.set_aspect('equal')
        self.ax.axhline(y=0, color='#6b7280', linewidth=0.8)
        self.ax.axvline(x=0, color='#6b7280', linewidth=0.8)
        self.ax.set_facecolor('#f9fafb')
        
        # Style the axes
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color('#9ca3af')
        self.ax.spines['left'].set_color('#9ca3af')
        
        # Set labels with clean styling
        self.ax.set_xlabel('X', fontsize=12, color='#374151')
        self.ax.set_ylabel('Y', fontsize=12, color='#374151')
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=(0, 20))
    
    def calculate_plot_bounds(self, original, transformed):
        """Calculate appropriate plot bounds based on data"""
        # Collect all x and y coordinates
        all_x = list(original[0]) if original is not None else []
        all_y = list(original[1]) if original is not None else []
        
        if transformed is not None:
            all_x.extend(transformed[0])
            all_y.extend(transformed[1])
        
        if not all_x or not all_y:
            return -3, 3, -3, 3  # Default bounds
        
        # Calculate bounds with padding
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Add 20% padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Ensure minimum range for small values
        x_range = max(x_range, 2.0)
        y_range = max(y_range, 2.0)
        
        x_padding = x_range * 0.2
        y_padding = y_range * 0.2
        
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        return x_min, x_max, y_min, y_max

    def update_display(self):
        """Update the visualization and UI elements"""
        # Get data for plotting
        original = self.engine.get_original_shape()
        transformed = self.engine.get_transformed_shape()
        
        # Calculate dynamic bounds
        x_min, x_max, y_min, y_max = self.calculate_plot_bounds(original, transformed)
        
        # Clear and reconfigure plot with dynamic bounds
        self.ax.clear()
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.grid(True, alpha=0.3, color='#d1d5db')
        self.ax.set_aspect('equal')
        self.ax.axhline(y=0, color='#6b7280', linewidth=0.8)
        self.ax.axvline(x=0, color='#6b7280', linewidth=0.8)
        self.ax.set_facecolor('#f9fafb')
        
        # Style the axes consistently
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color('#9ca3af')
        self.ax.spines['left'].set_color('#9ca3af')
        
        # Set labels
        self.ax.set_xlabel('X', fontsize=12, color='#374151')
        self.ax.set_ylabel('Y', fontsize=12, color='#374151')
        
        # Plot original shape
        if original is not None:
            self.ax.plot(original[0], original[1], color='#2563eb', linewidth=2.5, label='Original', alpha=0.8)
        
        # Plot transformed shape
        if transformed is not None:
            self.ax.plot(transformed[0], transformed[1], color='#dc2626', linewidth=2.5, label='Transformed', alpha=0.8)
        
        # Style the legend
        legend = self.ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.9)
        
        # Update the canvas
        self.canvas.draw()
        
        # Update matrix entries (only for 2x2 matrices)
        matrix = self.engine.get_current_matrix()
        if matrix.shape == (2, 2):
            self.a11.delete(0, 'end')
            self.a11.insert(0, f"{matrix[0,0]:.3f}")
            self.a12.delete(0, 'end')
            self.a12.insert(0, f"{matrix[0,1]:.3f}")
            self.a21.delete(0, 'end')
            self.a21.insert(0, f"{matrix[1,0]:.3f}")
            self.a22.delete(0, 'end')
            self.a22.insert(0, f"{matrix[1,1]:.3f}")
        else:
            # For non-2x2 matrices, clear the 2x2 input fields
            for entry in [self.a11, self.a12, self.a21, self.a22]:
                entry.delete(0, 'end')
                entry.insert(0, "N/A")
        
        # Update undo/redo buttons
        self.undo_btn.configure(state="normal" if self.engine.can_undo() else "disabled")
        self.redo_btn.configure(state="normal" if self.engine.can_redo() else "disabled")
        
        # Update info display
        self.update_info_display()
    
    def update_info_display(self):
        """Update the information display"""
        matrix = self.engine.get_current_matrix()
        
        info_text = f"Current Matrix ({matrix.shape[0]}x{matrix.shape[1]}):\n"
        
        # Format matrix display based on size
        if matrix.shape[0] <= 6 and matrix.shape[1] <= 6:
            info_text += f"{matrix}\n\n"
        else:
            info_text += f"Matrix too large to display ({matrix.shape[0]}x{matrix.shape[1]})\n\n"
        
        # Add matrix properties (only for square matrices)
        if matrix.shape[0] == matrix.shape[1]:
            det = self.engine.determinant()
            trace = self.engine.trace()
            info_text += f"Determinant: {det:.6f}\n"
            info_text += f"Trace: {trace:.6f}\n"
            info_text += f"Invertible: {'Yes' if abs(det) > 1e-10 else 'No'}\n\n"
        else:
            info_text += f"Matrix is {matrix.shape[0]}x{matrix.shape[1]} (non-square)\n"
            info_text += f"Determinant: N/A (non-square)\n"
            info_text += f"Trace: N/A (non-square)\n\n"
        
        history = self.engine.get_history()
        info_text += f"Operation History ({len(history)} operations):\n"
        for i, op in enumerate(history[-10:]):  # Show last 10 operations
            marker = " -> " if i == len(history[-10:]) - 1 else "    "
            info_text += f"{marker}{op}\n"
        
        self.info_text.configure(state="normal")
        self.info_text.delete("0.0", tk.END)
        self.info_text.insert("0.0", info_text)
        self.info_text.configure(state="disabled")

    # =====================================================
    # MATRIX CREATION METHODS
    # =====================================================
    
    def get_matrix_dimensions(self):
        """Get matrix dimensions from input fields"""
        try:
            rows = int(self.rows_entry.get())
            cols = int(self.cols_entry.get())
            if rows <= 0 or cols <= 0:
                raise ValueError("Dimensions must be positive")
            return rows, cols
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid dimensions: {e}")
            return None, None
    
    def create_identity(self):
        """Create identity matrix"""
        rows, cols = self.get_matrix_dimensions()
        if rows is not None and cols is not None:
            if rows != cols:
                messagebox.showerror("Error", "Identity matrix must be square (rows = cols)")
                return
            self.engine.create_identity_matrix(rows)
            self.update_display()
    
    def create_random(self):
        """Create random matrix"""
        rows, cols = self.get_matrix_dimensions()
        if rows is not None and cols is not None:
            self.engine.create_random_matrix(rows, cols, -5, 5)
            self.update_display()
    
    def create_zeros(self):
        """Create zero matrix"""
        rows, cols = self.get_matrix_dimensions()
        if rows is not None and cols is not None:
            self.engine.create_zero_matrix(rows, cols)
            self.update_display()
    
    def create_ones(self):
        """Create ones matrix"""
        rows, cols = self.get_matrix_dimensions()
        if rows is not None and cols is not None:
            self.engine.create_ones_matrix(rows, cols)
            self.update_display()
    
    def create_default_2x2(self):
        """Create default 2x2 matrix"""
        self.engine.set_matrix(np.array([[1, 2], [3, 4]]), "default_2x2")
        self.update_display()
    
    def create_6x2_example(self):
        """Create 6x2 example matrix for testing"""
        example_matrix = np.array([
            [1, 0],
            [0, 1],  
            [2, 3],
            [-1, 2],
            [0.5, -0.5],
            [3, -1]
        ])
        self.engine.set_matrix(example_matrix, "6x2_example")
        self.update_display()

    # =====================================================
    # TRANSFORMATION METHODS (UPDATED)
    # =====================================================
    
    def apply_rotation(self):
        """Apply rotation transformation based on selected mode"""
        try:
            angle = float(self.rotation_entry.get())
            if self.transform_mode.get() == "create":
                self.engine.create_rotation_matrix(angle)
            else:
                self.engine.apply_rotation(angle)
            self.update_display()
        except ValueError as e:
            messagebox.showerror("Error", f"Please enter a valid angle: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Transformation failed: {e}")
    
    def apply_scaling(self):
        """Apply scaling transformation based on selected mode"""
        try:
            scale_x = float(self.scale_x_entry.get())
            scale_y = float(self.scale_y_entry.get())
            if self.transform_mode.get() == "create":
                self.engine.create_scaling_matrix(scale_x, scale_y)
            else:
                self.engine.apply_scaling(scale_x, scale_y)
            self.update_display()
        except ValueError as e:
            messagebox.showerror("Error", f"Please enter valid scaling factors: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Transformation failed: {e}")
    
    def apply_shear(self):
        """Apply shear transformation based on selected mode"""
        try:
            shear_x = float(self.shear_x_entry.get())
            shear_y = float(self.shear_y_entry.get())
            if self.transform_mode.get() == "create":
                self.engine.create_shear_matrix(shear_x, shear_y)
            else:
                self.engine.apply_shear(shear_x, shear_y)
            self.update_display()
        except ValueError as e:
            messagebox.showerror("Error", f"Please enter valid shear factors: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Transformation failed: {e}")

    def apply_custom_matrix(self):
        """Apply the matrix from the input fields"""
        try:
            matrix = np.array([[float(self.a11.get()), float(self.a12.get())],
                              [float(self.a21.get()), float(self.a22.get())]])
            self.engine.set_matrix(matrix, "custom_matrix")
            self.update_display()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all matrix elements")
    
    def apply_reflection(self, axis):
        """Apply reflection transformation"""
        self.engine.create_reflection_matrix(axis)
        self.update_display()
    
    def transpose_matrix(self):
        """Transpose the current matrix"""
        self.engine.set_transpose_matrix()
        self.update_display()
    
    def inverse_matrix(self):
        """Compute the inverse of the current matrix"""
        if not self.engine.set_inverse_matrix():
            messagebox.showerror("Error", "Matrix is not invertible (determinant is zero)")
        else:
            self.update_display()
    
    def reset_matrix(self):
        """Reset to identity matrix"""
        self.engine.reset_to_identity()
        self.update_display()
    
    def scalar_multiply(self):
        """Multiply matrix by scalar"""
        try:
            scalar = float(self.scalar_entry.get())
            self.engine.scalar_multiplication(scalar)
            self.update_display()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid scalar value")
    
    def undo(self):
        """Undo last operation"""
        self.engine.undo()
        self.update_display()
    
    def redo(self):
        """Redo next operation"""
        self.engine.redo()
        self.update_display()
    
    def calculate_determinant(self):
        """Calculate and show determinant"""
        det = self.engine.determinant()
        messagebox.showinfo("Determinant", f"Determinant = {det:.6f}")
    
    def set_identity(self):
        """Set matrix entries to identity"""
        self.a11.delete(0, tk.END)
        self.a11.insert(0, "1")
        self.a12.delete(0, tk.END)
        self.a12.insert(0, "0")
        self.a21.delete(0, tk.END)
        self.a21.insert(0, "0")
        self.a22.delete(0, tk.END)
        self.a22.insert(0, "1")
        self.engine.reset_to_identity()
        self.update_display()
    
    def generate_random(self):
        """Generate random matrix values"""
        matrix = np.random.uniform(-2, 2, (2, 2))
        self.a11.delete(0, tk.END)
        self.a11.insert(0, f"{matrix[0,0]:.2f}")
        self.a12.delete(0, tk.END)
        self.a12.insert(0, f"{matrix[0,1]:.2f}")
        self.a21.delete(0, tk.END)
        self.a21.insert(0, f"{matrix[1,0]:.2f}")
        self.a22.delete(0, tk.END)
        self.a22.insert(0, f"{matrix[1,1]:.2f}")
        self.engine.set_matrix(matrix, "random_matrix")
        self.update_display()
    
    def show_dot_product(self):
        """Show dot product calculation demonstration"""
        matrix = self.engine.get_current_matrix()
        test_vector = np.array([1, 1])
        result = self.engine.dot_product(test_vector)
        
        msg = "Dot Product Demonstration:\n\n"
        msg += f"Matrix:\n{matrix}\n\n"
        msg += f"Test Vector: {test_vector}\n\n"
        msg += f"Result: {result}\n"
        
        messagebox.showinfo("Dot Product", msg)
    
    def clear_history(self):
        """Clear operation history"""
        self.engine.clear_history()
        self.update_display()

    def show_eigenvalues(self):
        """Show eigenvalues and eigenvectors"""
        eigenvals, eigenvecs = self.engine.eigenvalues_eigenvectors()
        
        msg = "Eigenvalues and Eigenvectors:\n\n"
        for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
            msg += f"Eigenvalue {i+1}: {val:.6f}\n"
            msg += f"Eigenvector {i+1}: [{vec[0]:.6f}, {vec[1]:.6f}]\n\n"
        
        messagebox.showinfo("Eigenvalue Decomposition", msg)
    
    def show_svd(self):
        """Show Singular Value Decomposition"""
        U, S, Vt = self.engine.singular_value_decomposition()
        
        msg = "Singular Value Decomposition:\n\n"
        msg += f"U matrix:\n{U}\n\n"
        msg += f"Singular values: {S}\n\n"
        msg += f"V^T matrix:\n{Vt}\n"
        
        messagebox.showinfo("Singular Value Decomposition", msg)
    
    def show_matrix_info(self):
        """Show comprehensive matrix information"""
        info = self.engine.get_matrix_info()
        
        msg = "Matrix Analysis:\n\n"
        msg += f"Determinant: {info['determinant']:.6f}\n"
        msg += f"Trace: {info['trace']:.6f}\n"
        msg += f"Rank: {info['rank']}\n"
        msg += f"Frobenius Norm: {info['norm_frobenius']:.6f}\n"
        msg += f"2-Norm: {info['norm_2']:.6f}\n\n"
        msg += f"Properties:\n"
        msg += f"- Invertible: {info['is_invertible']}\n"
        msg += f"- Symmetric: {info['is_symmetric']}\n"
        msg += f"- Orthogonal: {info['is_orthogonal']}\n"
        
        messagebox.showinfo("Matrix Information", msg)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = MatrixTransformationApp()
    app.run()
