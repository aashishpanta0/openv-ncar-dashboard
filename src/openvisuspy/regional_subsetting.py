import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.models import DataRange1d

class RegionalSubsetting:
    def __init__(self):
        # Create the figure for subsetting with placeholders for x and y axes labels
        self.subset_plot = figure(
            title="Regional Subsetting (Draw a region using `Selection tool` in main Image)",
            sizing_mode="stretch_both",
            x_axis_label="Longitude",
            y_axis_label="Latitude"
        )

        # Color mapping for the heatmap
        self.mapper = LinearColorMapper(palette="Turbo256", low=0, high=1)

        # Add a color bar for the values
        color_bar = ColorBar(color_mapper=self.mapper, location=(0, 0))
        self.subset_plot.add_layout(color_bar, 'right')

        # Prepare the panel view
        self.view = pn.Column(pn.pane.Bokeh(self.subset_plot, sizing_mode="stretch_both"))

        # Default placeholders for x, y, dw, dh
        self.x = 0
        self.y = 0
        self.dw = 1
        self.dh = 1

        # Track min and max values
        self.previous_min = None
        self.previous_max = None

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
        """Reset the view to its default state, clearing the plot and resetting color mapper."""
        self.subset_plot.renderers = []  # Remove all renderers (this clears the current image)
        self.mapper.low = 0  # Reset to default low value
        self.x=None
        self.y=None
        self.dw=None
        self.dh=None
        self.subset_plot.x_range = DataRange1d()  # Create new x_range
        self.subset_plot.y_range = DataRange1d()  # Create new y_range
        
        self.mapper.high = 1  # Reset to default high value
        self.previous_min = None  # Reset previous min
        self.previous_max = None  # Reset previous max

    def get_view(self):
        """Return the view for regional subsetting."""
        return self.view
