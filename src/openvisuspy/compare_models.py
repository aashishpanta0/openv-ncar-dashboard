import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.models import DataRange1d

class CompareModels:
    def __init__(self):
        self.subset_plot = figure(
            title="Select a Model from the dropdowns to plot the difference",
            sizing_mode="stretch_both",
            x_axis_label="Longitude",
            y_axis_label="Latitude"
        )

        self.mapper = LinearColorMapper(palette="Turbo256")

        # Add a color bar for the values
        color_bar = ColorBar(color_mapper=self.mapper, location=(0, 0))
        self.subset_plot.add_layout(color_bar, 'right')

        # Prepare the panel view
        self.view = pn.Column(pn.pane.Bokeh(self.subset_plot, sizing_mode="stretch_both"))

        self.x = 0
        self.y = 0
        self.dw = 1
        self.dh = 1

        # Track min and max values
        self.previous_min = None
        self.previous_max = None

    def set_data(self, climate_data1, climate_data2, x=None, y=None, dw=None, dh=None):
        """
        Set the lat/lon (x, y) range and the 2D climate data to be plotted.
        If x, y, dw, and dh are not provided, the previous values are used.
        climate_data: 2D array (rows x cols) containing the climate data
        x: Optional; x coordinate (longitude)
        y: Optional; y coordinate (latitude)
        dw: Optional; width of the data region (delta in x direction)
        dh: Optional; height of the data region (delta in y direction)
        """
        # Set x and y ranges based on provided input, or default to the global ranges
        self.x = -180 if x is None else x
        self.y = -80 if y is None else y
        self.dw = 360 if dw is None else dw  # Longitude range from -180 to 180
        self.dh = 170 if dh is None else dh / 2  # Latitude range from -80 to 90, reduced height by half

        # Calculate the difference between the two datasets
        diff_data = abs(climate_data1 - climate_data2)
        
        # Update the plot
        if len(self.subset_plot.renderers) > 0:
            image_renderer = self.subset_plot.renderers[0]
            self.subset_plot.match_aspect = True
            image_renderer.data_source.data = {
                'image': [diff_data],
                'x': [self.x],
                'y': [self.y],
                'dw': [self.dw],
                'dh': [self.dh]
            }
        else:
            # Update the x and y ranges for longitude and latitude
            self.subset_plot.x_range = DataRange1d(start=-180, end=180)
            self.subset_plot.y_range = DataRange1d(start=-80, end=90)
            self.subset_plot.match_aspect = True
            self.subset_plot.image(
                image=[diff_data], 
                x=self.x, y=self.y, dw=self.dw, dh=self.dh, 
                color_mapper=self.mapper
            )

    def reset_view(self):
        """Reset the view to its default state, clearing the plot and resetting color mapper."""
        self.subset_plot.renderers = []  
        self.mapper.low = 0  
        self.x=None
        self.y=None
        self.dw=None
        self.dh=None
        self.subset_plot.x_range = DataRange1d()  
        self.subset_plot.y_range = DataRange1d()  
        
        self.mapper.high = 1 
        self.previous_min = None  
        self.previous_max = None  

    def get_view(self):
        """Return the view for regional subsetting."""
        return self.view
