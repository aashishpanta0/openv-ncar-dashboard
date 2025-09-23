import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColorBar, LinearColorMapper, DataRange1d
from bokeh.palettes import Turbo256  # <-- use the actual palette list

class RegionalSubsetting:
    def __init__(self):
        # Subset figure
        self.subset_plot = figure(
            title="Regional Subsetting (Draw a region using `Selection tool` in main Image)",
            sizing_mode="stretch_both",
            x_axis_label="Longitude",
            y_axis_label="Latitude"
        )

        # Color mapping for the heatmap
        self.mapper = LinearColorMapper(palette=Turbo256, low=0, high=1)

        # Color bar
        self.color_bar = ColorBar(color_mapper=self.mapper, location=(0, 0))
        self.subset_plot.add_layout(self.color_bar, 'right')

        # Panel view
        self.view = pn.Column(pn.pane.Bokeh(self.subset_plot, sizing_mode="stretch_both"))

        # Stored region parameters
        self.x = 0.0
        self.y = 0.0
        self.dw = 1.0
        self.dh = 1.0

        # Track min/max across calls for stable color limits
        self.previous_min = None
        self.previous_max = None

        # Keep a handle to the image renderer (created on first call)
        self.image_renderer = None

    def set_latlon(self, climate_data, x=None, y=None, dw=None, dh=None):
        """
        Update the region + image. Returns a dict with region & stats so callers
        (e.g., slice.py) can consume details immediately.
        """
        # Update stored region parameters if provided
        if x is not None:  self.x = float(x)
        if y is not None:  self.y = float(y)
        if dw is not None: self.dw = float(dw)
        if dh is not None: self.dh = float(dh)

        # Ensure 2D float array; handle empty or all-NaN safely
        arr = np.asarray(climate_data, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"climate_data must be 2D, got shape {arr.shape}")

        # Compute new min/max robustly
        if np.all(np.isnan(arr)):
            new_min, new_max = 0.0, 0.0
        else:
            new_min = float(np.nanmin(arr))
            new_max = float(np.nanmax(arr))
            # If the tile is constant, widen a touch to avoid zero-range mapper
            if new_min == new_max:
                pad = 1e-12 if new_min == 0 else abs(new_min) * 1e-6
                new_min -= pad
                new_max += pad

        # Initialize/extend global color limits
        if self.previous_min is None or self.previous_max is None:
            self.previous_min, self.previous_max = new_min, new_max
        self.mapper.low = min(new_min, self.previous_min)
        self.mapper.high = max(new_max, self.previous_max)
        self.previous_min = self.mapper.low
        self.previous_max = self.mapper.high

        # Create or update the image glyph
        if self.image_renderer is None:
            self.image_renderer = self.subset_plot.image(
                image=[arr],
                x=self.x, y=self.y, dw=self.dw, dh=self.dh,
                color_mapper=self.mapper
            )
        else:
            self.image_renderer.data_source.data = {
                'image': [arr],
                'x': [self.x], 'y': [self.y], 'dw': [self.dw], 'dh': [self.dh]
            }

        # Let ranges auto-fit the provided bbox
        self.subset_plot.x_range = DataRange1d()
        self.subset_plot.y_range = DataRange1d()

        # Return details so callers can use them immediately
        return {
            "bbox": {"x": self.x, "y": self.y, "dw": self.dw, "dh": self.dh},
            "stats": {"min": float(new_min), "max": float(new_max)},
            "mapper": {"low": float(self.mapper.low), "high": float(self.mapper.high)},
            "shape": arr.shape,
            "all_nan": bool(np.all(np.isnan(arr)))
        }

    def reset_view(self):
        """Clear the image and reset color scaling/state (keep colorbar)."""
        if self.image_renderer is not None:
            try:
                self.subset_plot.renderers.remove(self.image_renderer)
            except Exception:
                pass
            self.image_renderer = None

        self.mapper.low = 0.0
        self.mapper.high = 1.0
        self.previous_min = None
        self.previous_max = None

        self.x = self.y = self.dw = self.dh = None
        self.subset_plot.x_range = DataRange1d()
        self.subset_plot.y_range = DataRange1d()

    def get_view(self):
        """Return the Panel view for embedding."""
        return self.view

