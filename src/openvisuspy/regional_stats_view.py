import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
import panel as pn

class RegionalStatsView:
    def __init__(self):
        self.source_general = ColumnDataSource(data=dict(x=[], y=[], label=[]))
        self.source_avg = ColumnDataSource(data=dict(x=[], y=[]))
        self.zonal_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.hist_source = ColumnDataSource(data=dict(top=[], left=[], right=[]))
        self.cdf_source = ColumnDataSource(data=dict(x=[], y=[]))
        categories = ["Min", "Avg", "Max"]
        self.stats_plot = figure(title="General Stats",
                                 x_range=categories,
                                 sizing_mode="stretch_both",
                                 tools="hover",
                                 tooltips=[("Value", "@y"), ("Type", "@label")],
                                 y_range=(-20, 140))
        self.stats_plot.line(x='x', y='y', source=self.source_general, line_width=2, color='blue')
        self.stats_plot.circle(x='x', y='y', source=self.source_general, size=10, color='red')
        self.stats_plot.legend.visible = False

        self.avg_plot = figure(title="Average Over Time",
                               sizing_mode="stretch_both",
                               y_range=(-20, 140),
                               x_axis_label='Time', y_axis_label='Average Value')
        self.avg_plot.line(x='x', y='y', source=self.source_avg, line_width=2, color='purple', legend_label="Avg Over Time")
        self.avg_plot.circle(x='x', y='y', source=self.source_avg, size=10, color='purple', legend_label="Avg Over Time")
        hover_avg = HoverTool(tooltips=[("Time", "@x"), ("Average", "@y")])
        self.avg_plot.add_tools(hover_avg)

        self.zonal_plot = figure(title="Zonal Average",
                                 sizing_mode="stretch_both",
                                 x_axis_label="Latitude", y_axis_label="Average Value",
                                 x_range=(0, 360))
        self.zonal_plot.line(x='x', y='y', source=self.zonal_source, line_width=2, color='orange')
        zonal_hover = HoverTool(tooltips=[("Latitude", "@x"), ("Average", "@y")])
        self.zonal_plot.add_tools(zonal_hover)

        self.hist_plot = figure(title="Value Distribution",
                                sizing_mode="stretch_both",
                                x_axis_label="Value", y_axis_label="Frequency")
        self.hist_plot.quad(top='top', bottom=0, left='left', right='right',
                            source=self.hist_source, fill_color="navy", alpha=0.5)

        self.cdf_plot = figure(title="Cumulative Distribution Function",
                               sizing_mode="stretch_both",
                               x_axis_label="Value", y_axis_label="Cumulative Probability")
        self.cdf_plot.line(x='x', y='y', source=self.cdf_source, line_width=2, color='darkgreen')
        cdf_hover = HoverTool(tooltips=[("Value", "@x"), ("Cumulative", "@y")])
        self.cdf_plot.add_tools(cdf_hover)

        self.dropdown = pn.widgets.Select(name="Select Visualization", 
                                          options=["General Stats", "Average Over Time", 
                                                   "Zonal Average", "Histogram", "CDF"],
                                          value="General Stats")
        self.dropdown.param.watch(self.update_plot_based_on_selection, 'value')

        self.current_model_data = None    
        self.multi_model_data = None      
        self.time_step = 0  
        self.first_average = None

        self.view = pn.Column(self.dropdown, sizing_mode="stretch_both")
        self.view.append(self.get_plot_view())

    def set_data(self, current_model_data, multi_model_data=None):
        self.current_model_data = current_model_data
        self.multi_model_data = multi_model_data
        self.update_plot_based_on_selection()
        avg_value = np.nanmean(self.current_model_data)
        if self.first_average is None:
            self.first_average = avg_value
            self.avg_plot.y_range.start = self.first_average - 10
            self.avg_plot.y_range.end = self.first_average + 10
        self.time_step += 1
        current_y_min, current_y_max = self.avg_plot.y_range.start, self.avg_plot.y_range.end
        self.avg_plot.y_range.start = min(avg_value - 10, current_y_min)
        self.avg_plot.y_range.end = max(avg_value + 10, current_y_max)
        new_data = dict(x=[self.time_step], y=[avg_value])
        self.source_avg.stream(new_data)
        vis_type = self.dropdown.value
        if vis_type == "Zonal Average":
            self.update_zonal_average()
        elif vis_type == "Histogram":
            self.update_histogram()
        elif vis_type == "CDF":
            self.update_cdf()

    def calculate_general_stats(self):
        if self.current_model_data is None:
            return None
        min_value = np.nanmin(self.current_model_data)
        max_value = np.nanmax(self.current_model_data)
        avg_value = np.nanmean(self.current_model_data)
        return min_value, max_value, avg_value

    def update_zonal_average(self):
        if self.current_model_data is None:
            return
        zonal = np.nanmean(self.current_model_data, axis=1)
        latitudes = np.linspace(0, 360, len(zonal))
        self.zonal_source.data = dict(x=latitudes, y=zonal)

    def update_histogram(self):
        if self.current_model_data is None:
            return
        values = self.current_model_data.flatten()
        values = values[~np.isnan(values)]
        if values.size == 0:
            self.hist_source.data = dict(top=[], left=[], right=[])
            return
        hist, edges = np.histogram(values, bins=20)
        self.hist_source.data = dict(top=hist.tolist(), left=edges[:-1].tolist(), right=edges[1:].tolist())

    def update_cdf(self):
        if self.current_model_data is None:
            return
        values = self.current_model_data.flatten()
        values = values[~np.isnan(values)]
        if values.size == 0:
            self.cdf_source.data = dict(x=[], y=[])
            return
        sorted_vals = np.sort(values)
        cumulative = np.linspace(0, 1, len(sorted_vals))
        self.cdf_source.data = dict(x=sorted_vals.tolist(), y=cumulative.tolist())

    def update_plot_based_on_selection(self, event=None):
        vis_type = self.dropdown.value
        self.view[-1] = self.get_plot_view()
        if vis_type == "General Stats":
            stats = self.calculate_general_stats()
            if stats is not None:
                min_val, max_val, avg_val = stats
                current_y_min, current_y_max = self.stats_plot.y_range.start, self.stats_plot.y_range.end
                new_y_min = min(min_val - 10, current_y_min)
                new_y_max = max(max_val + 10, current_y_max)
                self.stats_plot.y_range.start = new_y_min
                self.stats_plot.y_range.end = new_y_max
                self.source_general.data = dict(x=["Min", "Avg", "Max"], y=[min_val, avg_val, max_val], label=["Min", "Avg", "Max"])
        elif vis_type == "Average Over Time":
            pass
        elif vis_type == "Zonal Average":
            self.update_zonal_average()
        elif vis_type == "Histogram":
            self.update_histogram()
        elif vis_type == "CDF":
            self.update_cdf()

    def get_plot_view(self):
        vis_type = self.dropdown.value
        if vis_type == "General Stats":
            return pn.pane.Bokeh(self.stats_plot, sizing_mode="stretch_both")
        elif vis_type == "Average Over Time":
            return pn.pane.Bokeh(self.avg_plot, sizing_mode="stretch_both")
        elif vis_type == "Zonal Average":
            return pn.pane.Bokeh(self.zonal_plot, sizing_mode="stretch_both")
        elif vis_type == "Histogram":
            return pn.pane.Bokeh(self.hist_plot, sizing_mode="stretch_both")
        elif vis_type == "CDF":
            return pn.pane.Bokeh(self.cdf_plot, sizing_mode="stretch_both")
        else:
            return pn.pane.Markdown("Selected visualization is not implemented.")

    def get_view(self):
        return self.view

    def reset_avg_over_time(self):
        self.source_avg.data = dict(x=[], y=[])  
        self.time_step = 0
        self.first_average = None
