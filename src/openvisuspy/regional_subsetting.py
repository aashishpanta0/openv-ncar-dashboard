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
        Set the lat/lon (x, y) range and the 2D climate data to be plotted.
        If x, y, dw, and dh are not provided, the previous values are used.
        climate_data: 2D array (rows x cols) containing the climate data
        x: Optional; x coordinate (longitude)
        y: Optional; y coordinate (latitude)
        dw: Optional; width of the data region (delta in x direction)
        dh: Optional; height of the data region (delta in y direction)
        """
        # Update stored x, y, dw, and dh if provided, otherwise use previous values
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if dw is not None:
            self.dw = dw
        if dh is not None:
            self.dh = dh

        # Calculate the new min and max for the data
        new_min = np.nanmin(climate_data)
        new_max = np.nanmax(climate_data)

        # If this is the first set, initialize previous min/max
        if self.previous_min is None or self.previous_max is None:
            self.previous_min = new_min
            self.previous_max = new_max

        # Update the color mapper using the min of new and previous values
        self.mapper.low = min(new_min, self.previous_min)
        self.mapper.high = max(new_max, self.previous_max)

        # Update the previous min and max for future calls
        self.previous_min = self.mapper.low
        self.previous_max = self.mapper.high

        if len(self.subset_plot.renderers) > 0:
            image_renderer = self.subset_plot.renderers[0]
            image_renderer.data_source.data = {
                'image': [climate_data],
                'x': [self.x],
                'y': [self.y],
                'dw': [self.dw],
                'dh': [self.dh]
            }
        else:
            # If no renderer exists yet, create a new image renderer
            self.subset_plot.image(
                image=[climate_data], 
                x=self.x, y=self.y, dw=self.dw, dh=self.dh, 
                color_mapper=self.mapper
            )
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

