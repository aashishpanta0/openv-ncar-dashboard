import os, sys, logging, copy, traceback, base64, asyncio, io, time, threading
import types
from urllib.parse import urlparse, urlencode
from datetime import datetime, timedelta

import numpy as np
import panel as pn
from panel.layout import FloatPanel
from panel import Column, Row, GridBox, Card
from panel.pane import HTML, JSON, Bokeh

import bokeh
import bokeh.models
import bokeh.events
import bokeh.plotting
import bokeh.models.callbacks
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper, HoverTool, CustomJS

import param

from .stats_view import StatsView
from .regional_stats_view import RegionalStatsView
from .regional_subsetting import RegionalSubsetting
from .ai_insights import AIInsightsLogic

from .utils   import *
from .backend import Aborted, LoadDataset, ExecuteBoxQuery, QueryNode

logger = logging.getLogger(__name__)
pn.extension(
    js_files={
        'html2canvas': 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js',
        'dom-to-image': 'https://cdnjs.cloudflare.com/ajax/libs/dom-to-image/3.3.0/dom-to-image.min.js',
        'modernScreenshot': 'https://unpkg.com/modern-screenshot'
    }
)

SLICE_ID = 0
EPSILON = 0.001

# ------------------------------------------------------------
# UI layout template (no models/scenarios; adds time_hour)
# ------------------------------------------------------------
DEFAULT_SHOW_OPTIONS = {
    "top": [
        ["scene", "time_year", "time_month", "time_day", "time_hour",
         "timestep_delta", "play_sec", "play_button", "palette", "color_mapper_type",
         "resolution", "view_dependent", "num_refinements", "show_probe"],
        ["take_screenshot_button", "range_mode", "range_min", "range_max",
         "toggle_button", "region_button", "region_stats_button"]
    ],
    "bottom": [
        ["request", "response"]
    ]
}

# ------------------------------------------------------------
# Leap-aware, hour-based time helpers matching your IDX scheme
#   raw = 1940*365*24 + days_since_1940*24 + hour + 1
# ------------------------------------------------------------
BASE_YEAR = 1940

def calculate_raw_time(year: int, month: int, day: int, hour: int = 0) -> int:
    base = datetime(BASE_YEAR, 1, 1, 0, 0, 0)
    t    = datetime(year, month, day, hour, 0, 0)
    days_since = (t.date() - base.date()).days  # leap-aware
    return BASE_YEAR * 365 * 24 + days_since * 24 + hour + 1  # 1-based

def reverse_calculate_time(raw_time: int):
    raw0 = int(raw_time) - (BASE_YEAR * 365 * 24 + 1)
    days_since, hour = divmod(raw0, 24)
    dt = datetime(BASE_YEAR, 1, 1, 0, 0, 0) + timedelta(days=days_since, hours=hour)
    return dt.year, dt.month, dt.day, dt.hour, dt.date()

def get_actual_time(rawtime: int):
    _, _, _, _, d = reverse_calculate_time(int(rawtime))
    return d

# ------------------------------------------------------------
class ViewportUpdate: 
    pass

class Canvas:
    def __init__(self, id):
        self.id=id
        self.fig=None
        self.pdim=2

        self.events={
            bokeh.events.Tap: [],
            bokeh.events.DoubleTap: [],
            bokeh.events.SelectionGeometry: [],
            ViewportUpdate: []
        }

        self.fig_layout=Row(sizing_mode="stretch_both")	
        self.createFigure() 

        self.last_W=0
        self.last_H=0
        self.last_viewport=None
        self.setViewport([0,0,256,256])
		
    def onIdle(self):
        W,H=self.getWidth(),self.getHeight()
        if W==0 or H==0:  
            return

        x=self.fig.x_range.start
        w=self.fig.x_range.end-x
        y=self.fig.y_range.start
        h=self.fig.y_range.end-y

        if [x,y,w,h]==self.last_viewport and [self.last_W,self.last_H]==[W,H]:
            return

        if self.pdim==2 and [self.last_W,self.last_H]!=[W,H]:
            x+=0.5*w
            y+=0.5*h
            if (w/W) > (h/H): 
                h=w*(H/W) 
            else: 
                w=h*(W/H)
            x-=0.5*w
            y-=0.5*h

        self.last_W=W
        self.last_H=H
        self.last_viewport=[x,y,w,h]

        if not all([
            self.fig.x_range.start==x, self.fig.x_range.end==x+w,
            self.fig.y_range.start==y, self.fig.y_range.end==y+h
        ]):
            self.fig.x_range.start, self.fig.x_range.end = x,x+w
            self.fig.y_range.start, self.fig.y_range.end = y,y+h

        [fn(None) for fn in self.events[ViewportUpdate]]

    def on_event(self, evt, callback):
        self.events[evt].append(callback)

    def createFigure(self):
        old=self.fig

        self.pan_tool               = bokeh.models.PanTool()
        self.wheel_zoom_tool        = bokeh.models.WheelZoomTool()
        self.box_select_tool        = bokeh.models.BoxSelectTool()
        self.box_select_tool_helper = bokeh.models.TextInput()
        self.reset_fig              = bokeh.models.ResetTool()
        self.hover_avg              = HoverTool(tooltips=[("Longitude", "$x"), ("Latitude", "$y")])

        self.fig=bokeh.plotting.figure(tools=[self.pan_tool,self.reset_fig,self.wheel_zoom_tool,self.box_select_tool,self.hover_avg]) 
        self.fig.toolbar_location="right" 
        self.fig.toolbar.active_scroll = self.wheel_zoom_tool
        self.fig.toolbar.active_drag    = self.pan_tool
        self.fig.toolbar.active_inspect = None
        self.fig.toolbar.active_tap     = None

        self.fig.x_range = bokeh.models.Range1d(0,512) if old is None else old.x_range
        self.fig.y_range = bokeh.models.Range1d(0,512) if old is None else old.y_range
        self.fig.sizing_mode = 'stretch_both'          if old is None else old.sizing_mode
        self.fig.yaxis.axis_label  = "Latitude"        if old is None else old.xaxis.axis_label
        self.fig.xaxis.axis_label  = "Longitude"       if old is None else old.yaxis.axis_label
        self.fig.on_event(bokeh.events.Tap      , lambda evt: [fn(evt) for fn in self.events[bokeh.events.Tap      ]])
        self.fig.on_event(bokeh.events.DoubleTap, lambda evt: [fn(evt) for fn in self.events[bokeh.events.DoubleTap]])

        self.fig_layout[:]=[]
        self.fig_layout.append(Bokeh(self.fig))
        self.enableSelection()
        self.last_renderer={}

    def enableSelection(self,use_python_events=False):
        if use_python_events:
            self.fig.on_event(bokeh.events.SelectionGeometry, lambda s: print("Selection (python)"))
        else:
            def handleSelectionGeometry(attr,old,new):
                j=json.loads(new)
                x,y=float(j["x0"]),float(j["y0"])
                w,h=float(j["x1"])-x,float(j["y1"])-y
                evt=types.SimpleNamespace()
                evt.new=[x,y,w,h]
                [fn(evt) for fn in self.events[bokeh.events.SelectionGeometry]]
                logger.info(f"HandleSeletionGeometry {evt}")

            self.box_select_tool_helper.on_change('value', handleSelectionGeometry)
            self.fig.js_on_event(bokeh.events.SelectionGeometry, bokeh.models.callbacks.CustomJS(
                args=dict(widget=self.box_select_tool_helper), 
                code="""
                    widget.value=JSON.stringify(cb_obj.geometry, undefined, 2);
                """
            ))

    def setAxisLabels(self,x,y):
        self.fig.xaxis.axis_label  = 'Longitude'
        self.fig.yaxis.axis_label  = 'Latitude'		

    def getWidth(self):
        try:
            return self.fig.inner_width
        except:
            return 0

    def getHeight(self):
        try:
            return self.fig.inner_height
        except:
            return 0

    def getViewport(self):
        x=self.fig.x_range.start
        y=self.fig.y_range.start
        w=self.fig.x_range.end-x
        h=self.fig.y_range.end-y
        return [x,y,w,h]

    def setViewport(self,value):
        x,y,w,h=value
        self.last_W,self.last_H=0,0
        self.fig.x_range.start, self.fig.x_range.end = x, x+w
        self.fig.y_range.start, self.fig.y_range.end = y, y+h

    def showData(self, data, viewport, color_bar=None):
        x,y,w,h=viewport

        if len(data.shape)==1:
            self.pdim=1
            self.wheel_zoom_tool.dimensions="width"
            vmin,vmax=np.nanmin(data),np.nanmax(data)
            self.fig.y_range.start=0.5*(vmin+vmax)-1.2*0.5*(vmax-vmin)
            self.fig.y_range.end  =0.5*(vmin+vmax)+1.2*0.5*(vmax-vmin)
            self.fig.renderers.clear()
            xs=np.arange(x,x+w,w/data.shape[0])
            ys=data
            self.fig.line(xs,ys)
        else:
            assert(len(data.shape) in [2,3])
            self.pdim=2
            self.wheel_zoom_tool.dimensions="both"
            img=ConvertDataForRendering(data)
            dtype=img.dtype
            if all([
                self.last_renderer.get("source",None) is not None,
                self.last_renderer.get("dtype",None)==dtype,
                self.last_renderer.get("color_bar",None)==color_bar
            ]):
                self.last_renderer["source"].data={"image":[img], "Longitude":[x], "Latitude":[y], "dw":[w], "dh":[h]}
            else:
                self.createFigure()
                source = bokeh.models.ColumnDataSource(data={"image":[img], "Longitude":[x], "Latitude":[y], "dw":[w], "dh":[h]})
                if img.dtype==np.uint32:	
                    self.fig.image_rgba("image", source=source, x="Longitude", y="Latitude", dw="dw", dh="dh") 
                else:
                    self.fig.image("image", source=source, x="Longitude", y="Latitude", dw="dw", dh="dh", color_mapper=color_bar.color_mapper) 
                self.fig.add_layout(color_bar, 'right')
                self.last_renderer={
                    "source": source,
                    "dtype":img.dtype,
                    "color_bar":color_bar
                }

# ------------------------------------------------------------
class Slice(param.Parameterized):
    def __init__(self):
        super().__init__()
        pn.extension(
            js_files={'html2canvas': 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js',
                      'dom-to-image':'https://cdnjs.cloudflare.com/ajax/libs/dom-to-image/3.3.0/dom-to-image.min.js',
                      'modernScreenshot': 'https://unpkg.com/modern-screenshot'}
        )
        self.render_id = pn.widgets.IntSlider(name="RenderId", value=0)

        # Scene & time
        self.scene_body = pn.widgets.TextAreaInput(name='Current', sizing_mode="stretch_width", height=520)
        self.scene = pn.widgets.Select(name="Scene", options=[], width=120)
        self.timestep = pn.widgets.IntSlider(name="Time", value=0, start=0, end=1, step=1, sizing_mode="stretch_width")
        self.time_year  = pn.widgets.Select(name="Year", options=[], sizing_mode="stretch_width")
        self.time_month = pn.widgets.Select(name="Month", options=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], sizing_mode="stretch_width")
        self.time_day   = pn.widgets.Select(name="Date", options=[i for i in range(1,32)], sizing_mode="stretch_width")
        self.time_hour  = pn.widgets.Select(name="Hour", options=[i for i in range(0,24)], sizing_mode="stretch_width")

        # Playback / field
        self.timestep_delta = pn.widgets.Select(name="Speed", options=[1, 2, 4, 8, 16, 32, 64, 128], value=1, width=50)
        self.field = pn.widgets.Select(name='Field', options=['2T'], value='2T', width=80)
        self.field.visible = False  # fixed field, hide if you want

        # View / query
        self.resolution = pn.widgets.IntSlider(name='Resolution', value=15, start=15, end=22, sizing_mode="stretch_width")
        self.view_dependent = pn.widgets.Select(name="ViewDep", options={"Yes": True, "No": False}, value=True, width=80)
        self.num_refinements = pn.widgets.IntSlider(name='#Ref', value=0, start=0, end=4, width=80)
        self.direction = pn.widgets.Select(name='Direction', options={'X': 0, 'Y': 1, 'Z': 2}, value=2, width=80)
        self.offset = pn.widgets.EditableFloatSlider(name="Depth", start=0.0, end=1024.0, step=1.0, value=0.0,
                                                     sizing_mode="stretch_width",
                                                     format=bokeh.models.formatters.NumeralTickFormatter(format="0.01"))
        self.viewport = pn.widgets.TextInput(name="Viewport", value="")
        self.toggle_button = pn.widgets.Button(name="Show Stats", button_type="success", width=100)
        self.region_button = pn.widgets.Button(name="Turn on Regional Setting", button_type="success", width=100)
        self.region_stats_button = pn.widgets.Button(name='Show Regional Stats', visible=False, margin=(5,10,10,50))

        # Extra features still supported
        self.generate_insights_button = pn.widgets.Button(name="Turn on AI Insights", button_type="success", width=100, margin=(5,10,10,70))

        # Palette / range
        self.range_mode = pn.widgets.Select(name="Range", options=["metadata", "user", "dynamic", "dynamic-acc"], value="user", width=120)
        self.range_min = pn.widgets.FloatInput(name="Min", width=80, value=0)
        self.range_max = pn.widgets.FloatInput(name="Max", width=80, value=100)
        self.palette = pn.widgets.ColorMap(name="Palette", options=GetPalettes(), value_name="Viridis", ncols=5, width=180)
        self.color_mapper_type = pn.widgets.Select(name="Mapper", options=["linear", "log"], width=60)

        # Transport & misc
        self.play_button = pn.widgets.Button(name="Play", width=10, sizing_mode='stretch_width')
        self.play_sec = pn.widgets.Select(name="Frame delay", options=[0.00, 0.01, 0.1, 0.2, 0.1, 1, 2], value=0.01, width=120)
        self.request = pn.widgets.TextInput(name="", sizing_mode='stretch_width', disabled=False)
        self.response = pn.widgets.TextInput(name="", sizing_mode='stretch_width', disabled=False)
        self.info_button = pn.widgets.Button(icon="info-circle", width=20)	
        self.open_button = pn.widgets.Button(icon="file-upload", width=20)
        self.save_button = pn.widgets.Button(icon="file-download", width=20)
        self.copy_url_button = pn.widgets.Button(icon="copy", width=20)
        self.take_screenshot_button = pn.widgets.Button(name='Take Screenshot', width=120, button_type='primary')
        self.logout_button = pn.widgets.Button(icon="logout", width=20)
        self.save_button_helper = pn.widgets.TextInput(visible=False)
        self.copy_url_button_helper = pn.widgets.TextInput(visible=False)
        self.take_screenshot_button_helper = pn.widgets.TextInput(visible=False)
        self.file_name_input = pn.widgets.TextInput(name="Numpy_File", value='test', placeholder='Numpy File Name to save')

        # Panels
        self.stats_panel = pn.Column(sizing_mode="stretch_both", visible=False)
        self.region_stats_panel = pn.Column(sizing_mode="stretch_both", visible=False)
        self.region_panel = pn.Column(sizing_mode="stretch_both", visible=False)
        self.generate_insights_panel = pn.Column(sizing_mode="stretch_both", visible=False)
        self.insight_text = pn.widgets.TextAreaInput(value="", height=200, width=400, disabled=True, visible=False)

        # Logic backends
        self.stats_view = StatsView()
        self.region_stats_view = RegionalStatsView()
        self.region_view = RegionalSubsetting()
        self.ai_insights = AIInsightsLogic()

        self.regional_setting_enabled = False
        self.ai_insights_enabled = False

        # Buttons: stats panel toggles
        def toggle_stats(event):
            if self.stats_panel.visible:
                self.stats_panel.visible = False
                self.toggle_button.name = "Show Stats"
                self.refresh()
            else:
                self.stats_panel.visible = True
                self.toggle_button.name = "Hide Stats"
                self.stats_panel[:] = [self.stats_view.get_view()]
        self.toggle_button.on_click(toggle_stats)

        def toggle_regional_stats(event):
            if self.region_stats_panel.visible:
                self.region_stats_panel.visible = False
                self.region_stats_button.name = "Show Regional Stats"
                self.refresh()
            else:
                self.region_stats_panel.visible = True
                self.region_stats_button.name = "Hide Regional Stats"
                self.region_stats_panel[:] = [self.region_stats_view.get_view()]
        self.region_stats_button.on_click(toggle_regional_stats)

        def toggle_ai_insights(event):
            if self.generate_insights_button.name == "Turn on AI Insights":
                insights = self.ai_insights.generate_insights()
                self.insight_text.value = insights
                self.insight_text.visible = True
                self.generate_insights_button.name = "Turn off AI Insights"
                self.generate_insights_button.button_type = "danger"
                self.ai_insights_enabled = True 
            else:
                self.insight_text.visible = False
                self.generate_insights_button.name = "Turn on AI Insights"
                self.generate_insights_button.button_type = "success"
                self.ai_insights_enabled = False
            self.main_layout[1] = self.build_middle_layout()
        self.generate_insights_button.on_click(toggle_ai_insights)

        def toggle_region(event):
            if self.region_panel.visible or self.region_stats_panel.visible:
                self.region_panel.visible = False
                self.region_stats_panel.visible = False
                self.region_button.name = "Turn on Regional Setting"
                self.region_button.button_type = "success"
                self.refresh()
            else:
                self.region_panel.visible = True
                self.region_stats_button.name = "Show Regional Stats"
                self.region_button.name = "Turn off Regional Setting"
                self.region_button.button_type = "danger"
                self.region_panel[:] = [self.region_view.get_view()]
        self.region_button.on_click(toggle_region)

        self.region_stats_button.param.watch(self.update_layout, 'value')

        # Palette/Range callbacks
        def onPaletteChange(evt):
            self.color_bar=None
            self.refresh()
        self.palette.param.watch(SafeCallback(onPaletteChange),"value_name", onlychanged=True,queued=True)

        def onRangeModeChange(evt):
            mode=evt.new
            self.color_map=None
            if mode == "metadata":   
                self.range_min.value = self.metadata_range[0]
                self.range_max.value = self.metadata_range[1]
            if mode == "dynamic-acc":
                self.range_min.value = 0.0
                self.range_max.value = 0.0
            self.range_min.disabled = (mode != "user")
            self.range_max.disabled = (mode != "user")
            self.refresh()
        self.range_mode.param.watch(SafeCallback(onRangeModeChange),"value", onlychanged=True,queued=True)

        def onRangeChange(evt):
            self.color_map=None
            self.color_bar=None
            self.refresh()
        self.range_min.param.watch(SafeCallback(onRangeChange),"value", onlychanged=True,queued=True)
        self.range_max.param.watch(SafeCallback(onRangeChange),"value", onlychanged=True,queued=True)

        def onColorMapperTypeChange(evt):
            self.color_bar=None 
            self.refresh()
        self.color_mapper_type.param.watch(SafeCallback(onColorMapperTypeChange),"value", onlychanged=True,queued=True)

        def onResChange(evt):
            self.view_dependent.value=False
            self.refresh()
        self.resolution.param.watch(SafeCallback(onResChange),"value", onlychanged=True,queued=True)

        def onDirectionChange(evt):
            value=evt.new
            logger.debug(f"id={self.id} value={value}")
            pdim = self.getPointDim()
            if pdim in (1,2): value = 2
            dims = [int(it) for it in self.db.getLogicSize()]
            offset_value,offset_range=self.guessOffset(value)
            self.offset.start=offset_range[0]
            self.offset.end  =offset_range[1]
            self.offset.step=1e-16 if self.offset.editable and offset_range[2]==0.0 else offset_range[2]
            self.offset.value=offset_value
            self.setQueryLogicBox(([0]*pdim,dims))
            self.refresh()
        self.direction.param.watch(SafeCallback(onDirectionChange),"value", onlychanged=True,queued=True)

        self.offset.param.watch(SafeCallback(lambda evt: self.refresh()),"value", onlychanged=True,queued=True)

        # Time wiring: slider <-> (Y/M/D/H)
        def onTimestepChange(evt):
            y, m, d, h, _ = reverse_calculate_time(self.timestep.value)
            self.time_year.value  = y
            self.time_month.value = datetime(2000, m, 1).strftime("%b")
            self.time_day.value   = d
            self.time_hour.value  = h
            self.refresh()
        self.timestep.param.watch(SafeCallback(onTimestepChange), "value", onlychanged=True, queued=True)

        def _recalc_timestep_from_widgets():
            mnum = datetime.strptime(self.time_month.value, "%b").month
            self.timestep.value = calculate_raw_time(self.time_year.value, mnum, self.time_day.value, self.time_hour.value)

        def onMonthChange(evt):
            days = {'Jan':31,'Feb':28,'Mar':31,'Apr':30,'May':31,'Jun':30,
                    'Jul':31,'Aug':31,'Sep':30,'Oct':31,'Nov':30,'Dec':31}
            self.time_day.options = list(range(1, days[self.time_month.value] + 1))
            if self.time_day.value > days[self.time_month.value]:
                self.time_day.value = days[self.time_month.value]
            _recalc_timestep_from_widgets(); self.refresh()
        self.time_month.param.watch(SafeCallback(onMonthChange), "value", onlychanged=True, queued=True)

        def onYearChange(evt):
            _recalc_timestep_from_widgets(); self.refresh()
        self.time_year.param.watch(SafeCallback(onYearChange), "value", onlychanged=True, queued=True)

        def onDayChange(evt):
            _recalc_timestep_from_widgets(); self.refresh()
        self.time_day.param.watch(SafeCallback(onDayChange), "value", onlychanged=True, queued=True)

        def onHourChange(evt):
            _recalc_timestep_from_widgets(); self.refresh()
        self.time_hour.param.watch(SafeCallback(onHourChange), "value", onlychanged=True, queued=True)

        # Toolbar actions
        self.info_button.on_click(SafeCallback(lambda evt: self.showInfo()))
        self.open_button.on_click(SafeCallback(lambda evt: self.showOpen()))
        self.save_button.on_click(SafeCallback(lambda evt: self.save()))
        self.copy_url_button.on_click(SafeCallback(lambda evt: self.copyUrl()))
        self.take_screenshot_button.on_click(SafeCallback(lambda evt: self.takeScreenshot()))
        self.play_button.on_click(SafeCallback(lambda evt: self.togglePlay()))

        # Internal state
        self.on_change_callbacks={}
        self.num_hold=0
        global SLICE_ID
        self.id=SLICE_ID
        SLICE_ID += 1
	
        self.db = None
        self.access = None
        self.detailed_data=None
        self.selected_physic_box=None
        self.selected_logic_box=None

        self.logic_to_physic        = [(0.0, 1.0)] * 3
        self.metadata_range         = [0.0, 255.0]
        self.scenes                 = {}

        self.scene_body.stylesheets=[""".bk-input {background-color: rgb(48, 48, 64);color: white;font-size: small;}"""]

        self.createGui()

        def onSceneChange(evt): 
            logger.info(f"onSceneChange {evt}")
            body=self.scenes[evt.new]
            self.setSceneBody(body)
        self.scene.param.watch(SafeCallback(onSceneChange),"value", onlychanged=True,queued=True)

    # ----- Layout helpers -----
    def build_middle_layout(self):
        if self.regional_setting_enabled:
            return Row(
                Column(self.middle_layout, self.stats_panel),
                Column(self.region_panel, self.region_stats_panel)
            )
        elif self.ai_insights_enabled:
            return Column(self.middle_layout, self.stats_panel, self.insight_text)
        else:
            return Column(self.middle_layout, self.stats_panel)

    def update_layout(self, event):
        self.main_layout[1] = self.build_middle_layout()

    # ----- Create GUI skeleton -----
    def createGui(self):

        self.save_button.js_on_click(args={"source":self.save_button_helper}, code="""
            function jsSave() {
                const link = document.createElement("a");
                const file = new Blob([source.value], { type: 'text/plain' });
                link.href = URL.createObjectURL(file);
                link.download = "save_scene.json";
                link.click();
                URL.revokeObjectURL(link.href);
            }
            setTimeout(jsSave,300);
        """)

        self.take_screenshot_button.js_on_click(args={"source":self.take_screenshot_button_helper}, code="""
        function ensureModernScreenshot(callback) {
            if (typeof modernScreenshot === 'undefined') {
                if (!document.getElementById('modernScreenshotScript')) {
                    var script = document.createElement('script');
                    script.id = 'modernScreenshotScript';
                    script.src = 'https://unpkg.com/modern-screenshot';
                    script.onload = function() { if (callback) callback(); };
                    document.head.appendChild(script);
                } else {
                    document.getElementById('modernScreenshotScript').onload = function() { if (callback) callback(); };
                }
            } else {
                callback();
            }
        }
        window.scrollTo(0, 0);
        ensureModernScreenshot(function() {
            setTimeout(function () {
                modernScreenshot.domToPng(document.body, { scale: 2 })
                .then(function (dataUrl) {
                    var a = document.createElement('a');
                    a.href = dataUrl;
                    a.download = 'dashboard_screenshot.png';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                })
                .catch(function (error) { console.error('Error capturing screenshot:', error); });
            }, 100);
        });
        """)

        self.copy_url_button.js_on_click(args={"source": self.copy_url_button_helper}, code="""
            function jsCopyUrl() { navigator.clipboard.writeText(source.value); } 
            setTimeout(jsCopyUrl,300);
        """)

        self.logout_button = pn.widgets.Button(icon="logout",width=20)
        self.logout_button.js_on_click(args={"source": self.logout_button}, code="""
            window.location = window.location.href + "/logout";
        """)

        # play control
        self.play = types.SimpleNamespace()
        self.play.is_playing = False
        self.idle_callback = None
        self.color_bar     = None
        self.query_node    = QueryNode()
        self.query_node2   = QueryNode()

        self.t1=time.time()
        self.aborted       = Aborted()
        self.new_job       = True
        self.current_img   = None
        self.last_job_pushed = time.time()

        self.canvas = Canvas(self.id)
        self.canvas.on_event(ViewportUpdate,              SafeCallback(self.onCanvasViewportChange))
        self.canvas.on_event(bokeh.events.Tap           , SafeCallback(self.onCanvasSingleTap))
        self.canvas.on_event(bokeh.events.DoubleTap     , SafeCallback(self.onCanvasDoubleTap))

        self.top_layout=Column(sizing_mode="stretch_width")
        self.middle_layout=Column(Row(self.canvas.fig_layout, sizing_mode='stretch_both'), sizing_mode='stretch_both')
        self.bottom_layout=Column(sizing_mode="stretch_width")

        self.dialogs=Column()
        self.dialogs.visible=False

        self.main_layout=Column(
            self.top_layout,
            self.build_middle_layout(),
            self.bottom_layout, 
            self.dialogs,
            self.copy_url_button_helper,
            self.take_screenshot_button_helper,
            self.save_button_helper,
            sizing_mode="stretch_both"
        )

        self.setShowOptions(DEFAULT_SHOW_OPTIONS)
        self.canvas.on_event(bokeh.events.SelectionGeometry, SafeCallback(self.showDetails))
        self.start()

    # ----- Misc UI plumbing -----
    def onCanvasViewportChange(self, evt):
        x,y,w,h=self.canvas.getViewport()
        self.viewport.value=f"{x} {y} {w} {h}"
        self.refresh()

    def onCanvasSingleTap(self, evt): pass
    def onCanvasDoubleTap(self, evt): pass

    def getShowOptions(self):
        return self.show_options

    def setShowOptions(self, value):
        self.show_options=value
        for layout, position in ((self.top_layout,"top"),(self.bottom_layout,"bottom")):
            layout.clear()
            for row in value.get(position,[[]]):
                v=[]
                for widget in row:
                    if isinstance(widget,str):
                        widget=getattr(self, widget.replace("-","_"),None)
                    if widget:
                        v.append(widget)
                if v: layout.append(Row(*v,sizing_mode="stretch_width"))

    def getShareableUrl(self):
        body=self.getSceneBody()
        load_s=base64.b64encode(json.dumps(body).encode('utf-8')).decode('ascii')
        current_url=GetCurrentUrl()
        o=urlparse(current_url)
        return o.scheme + "://" + o.netloc + o.path + '?' + urlencode({'load': load_s})

    def stop(self):
        self.aborted.setTrue()
        self.query_node.stop()
        self.query_node2.stop()

    def start(self):
        self.query_node.start()
        self.query_node2.start()
        if not self.idle_callback:
            self.idle_callback = AddPeriodicCallback(self.onIdle, 1000 // 30)
        self.refresh()

    def getMainLayout(self):
        return self.main_layout

    def getLogicToPhysic(self):
        return self.logic_to_physic

    def setLogicToPhysic(self, value):
        logger.debug(f"id={self.id} value={value}")
        self.logic_to_physic = value
        self.refresh()

    def getPhysicBox(self):
        dims = self.db.getLogicSize()
        vt = [it[0] for it in self.logic_to_physic]
        vs = [it[1] for it in self.logic_to_physic]
        return [[0 * vs[I] + vt[I], dims[I] * vs[I] + vt[I]] for I in range(len(dims))]

    def setPhysicBox(self, value):
        dims = self.db.getLogicSize()
        def LinearMapping(a, b, A, B):
            vs = (B - A) / (b - a)
            vt = A - a * vs
            return vt, vs
        T = [LinearMapping(0, dims[I], *value[I]) for I in range(len(dims))]
        self.setLogicToPhysic(T)
		
    def getSceneBody(self):
        return {
            "scene" : {
                "name": self.scene.value, 
                "timestep-delta": self.timestep_delta.value,
                "timestep": self.timestep.value,
                "direction": self.direction.value,
                "offset": self.offset.value, 
                "field": self.field.value,   # always '2T'
                "view-dependent": self.view_dependent.value,
                "resolution": self.resolution.value,
                "num-refinements": self.num_refinements.value,
                "play-sec": self.play_sec.value,
                "palette": self.palette.value_name,
                "color-mapper-type": self.color_mapper_type.value,
                "range-mode": self.range_mode.value,
                "range-min": cdouble(self.range_min.value),
                "range-max": cdouble(self.range_max.value),
                "viewport": self.canvas.getViewport()
            }
        }

    def hold(self):
        self.num_hold=getattr(self,"num_hold",0) + 1

    def unhold(self):
        self.num_hold-=1

    # ----- Scene load -----
    def load(self, value):
        if isinstance(value,str):
            ext=os.path.splitext(value)[1].split("?")[0]
            if ext==".json":
                value=LoadJSON(value)
            else:
                value={"scenes": [{"name": os.path.basename(value), "url":value}]}
        elif isinstance(value,dict):
            pass
        else:
            raise Exception(f"{value} not supported")

        assert(isinstance(value,dict))
        assert(len(value)==1)
        root=list(value.keys())[0]

        self.scenes={}
        for it in value[root]:
            if "name" in it:
                self.scenes[it["name"]]={"scene": it}

        self.scene.options = list(self.scenes)
        if self.scenes:
            first_scene_name=list(self.scenes)[0]
            self.setSceneBody(self.scenes[first_scene_name])

    def setSceneBody(self, scene):
        logger.info(f"# //////////////////////////////////////////#")
        logger.info(f"id={self.id} {scene} START")

        assert(isinstance(scene,dict))
        assert(len(scene)==1 and list(scene.keys())==["scene"])
        scene=scene["scene"]

        name=scene["name"]
        assert(name in self.scenes)
        default_scene=self.scenes[name]["scene"]
        url = default_scene["url"]
        urls = default_scene.get("urls",{})

        if "urls" in scene:
            locals=[it for it in urls if it.get('id')=="local"]
            if locals and os.path.isfile(locals[0]["url"]):
                logger.info(f"id={self.id} Overriding url from {locals[0]['url']} since it exists and is local")
                url = locals[0]["url"]

        logger.info(f"id={self.id} LoadDataset url={url}...")
        db=LoadDataset(url=url) 
        self.data_url=url
        self.db    = db
        self.access=db.createAccess()
        self.scene.value=name

        # Fixed field
        self.field.value = "2T"
        db_field = self.db.getField(self.field.value)
        self.metadata_range = [db_field.getDTypeRange().From, db_field.getDTypeRange().To]

        # Timesteps: use full dataset range
        all_timesteps=self.db.getTimesteps()
        self.timestep.start = int(all_timesteps[0])
        self.timestep.end   = int(all_timesteps[-1])
        self.timestep.step  = 1

        # Initialize Y/M/D/H from current timestep
        y, m, d, h, _ = reverse_calculate_time(int(scene.get("timestep", self.db.getTimesteps()[0])))
        self.time_year.options = list(range(y, y+1))  # set properly below once we decode bounds
        # Better: compute min/max year from endpoints
        y0, m0, d0, h0, _ = reverse_calculate_time(self.timestep.start)
        y1, m1, d1, h1, _ = reverse_calculate_time(self.timestep.end)
        self.time_year.options = list(range(y0, y1+1))

        self.time_year.value  = y
        self.time_month.value = datetime(2000, m, 1).strftime("%b")
        self.time_day.value   = d
        self.time_hour.value  = h
        self.timestep.value   = int(scene.get("timestep", self.db.getTimesteps()[0]))

        # Logic/physic mapping
        pdim = self.getPointDim()
        if "logic-to-physic" in scene:
            logic_to_physic=scene["logic-to-physic"]
            self.setLogicToPhysic(logic_to_physic)
        else:
            physic_box = self.db.inner.idxfile.bounds.toAxisAlignedBox().toString().strip().split()
            physic_box = [(float(physic_box[I]), float(physic_box[I + 1])) for I in range(0, pdim * 2, 2)]
            self.setPhysicBox(physic_box)

        # Directions
        directions = self.db.inner.idxfile.axis.strip().split()
        directions = {it: I for I, it in enumerate(directions)} if directions else  {'X':0,'Y':1,'Z':2}
        self.direction.options=directions

        # Other scene params
        self.timestep_delta.value=int(scene.get("timestep-delta", 1))
        self.view_dependent.value = bool(scene.get('view-dependent', True))

        resolution=int(scene.get("resolution", -6))
        if resolution<0: resolution=self.db.getMaxResolution()+resolution
        self.resolution.end = self.db.getMaxResolution()
        self.resolution.value = resolution

        self.range_mode.value="user"
        self.range_min.value=self.metadata_range[0]
        self.range_max.value=self.metadata_range[1]

        self.num_refinements.value=int(scene.get("num-refinements", 1))
        self.direction.value = int(scene.get("direction", 2))

        default_offset_value,offset_range=self.guessOffset(self.direction.value)
        self.offset.start=offset_range[0]
        self.offset.end  =offset_range[1]
        self.offset.step=1e-16 if self.offset.editable and offset_range[2]==0.0 else offset_range[2]
        self.offset.value=float(scene.get("offset",default_offset_value))
        self.setQueryLogicBox(([0]*self.getPointDim(),[int(it) for it in self.db.getLogicSize()]))

        self.play_sec.value=float(scene.get("play-sec",0.01))
        self.palette.value_name=scene.get("palette",DEFAULT_PALETTE)
        self.color_mapper_type.value = scene.get("color-mapper-type","linear")	

        viewport=scene.get("viewport",None)
        if viewport is not None:
            self.canvas.setViewport(viewport)

        show_options=scene.get("show-options",DEFAULT_SHOW_OPTIONS)
        self.setShowOptions(show_options)
        self.start()
        logger.info(f"id={self.id} END\n")

    # ----- Info / Open / Save / URL -----
    def showInfo(self):
        body=self.scenes[self.scene.value]
        metadata=body["scene"].get("metadata", [])
        cards=[]
        for I, item in enumerate(metadata):
            type = item["type"]
            filename = item.get("filename",f"metadata_{I:02d}.bin")
            if type == "b64encode":
                body_b = base64.b64decode(item["encoded"]).decode("utf-8")
                internal_panel=HTML(f"<div><pre><code>{body_b}</code></pre></div>",sizing_mode="stretch_width",height=400)
            elif type=="json-object":
                obj=item["object"]
                internal_panel=JSON(obj,name="Object",depth=3, sizing_mode="stretch_width",height=400) 
            else:
                continue
            cards.append(Card(
                internal_panel,
                pn.widgets.FileDownload(io.StringIO(body_b if type=="b64encode" else json.dumps(obj)),
                                        embed=True, filename=filename, align="end"),
                title=filename, collapsed=(I>0), sizing_mode="stretch_width"
            ))
        self.showDialog(*cards)

    def showOpen(self):
        def onLoadClick(evt):
            body=value.decode('ascii')
            self.scene_body.value=body
            ShowInfoNotification('Load done. Press `Eval`')
        file_input = pn.widgets.FileInput(description="Load", accept=".json")
        file_input.param.watch(SafeCallback(onLoadClick),"value", onlychanged=True,queued=True)

        def onEvalClick(evt):
            self.setSceneBody(json.loads(self.scene_body.value))
            ShowInfoNotification('Eval done')
        eval_button = pn.widgets.Button(name="Eval", align='end')
        eval_button.on_click(SafeCallback(onEvalClick))

        self.showDialog(
            Column(
                self.scene_body,
                Row(file_input, eval_button, align='end'),
                sizing_mode="stretch_both",align="end"
            ), 
            width=600, height=700, name="Open")

    def save(self):
        body=json.dumps(self.getSceneBody(),indent=2)
        self.save_button_helper.value=body
        ShowInfoNotification('Save done')
        print(body)

    def copyUrl(self):
        self.copy_url_button_helper.value=self.getShareableUrl()
        ShowInfoNotification('Copy url done')

    def takeScreenshot(self):
        self.take_screenshot_button_helper.value=""
        ShowInfoNotification('Taking Screenshot')

    # ----- Dialog helper -----
    def showDialog(self, *args, **kwargs):
        d={"position":"center", "width":1024, "height":600, "contained":False}
        d.update(**kwargs)
        float_panel=FloatPanel(*args, **d)
        self.dialogs.append(float_panel)

    # ----- Dataset helpers -----
    def getMaxResolution(self):
        return self.db.getMaxResolution()

    def setViewDependent(self, value):
        logger.debug(f"id={self.id} value={value}")
        self.view_dependent.value = value
        self.refresh()

    def getLogicAxis(self):
        dir  = self.direction.value
        directions = self.direction.options
        XY = list(directions.values())
        if len(XY) == 3:
            del XY[dir]
        else:
            assert (len(XY) == 2)
        X, Y = XY
        Z = dir if len(directions) == 3 else 2
        titles = list(directions.keys())
        return (X, Y, Z), (titles[X], titles[Y], titles[Z] if len(titles) == 3 else 'Z')

    def guessOffset(self, dir):
        pdim = self.getPointDim()
        if pdim<=2:
            return 0, [0, 0, 1]
        else:
            vt = [self.logic_to_physic[I][0] for I in range(pdim)]
            vs = [self.logic_to_physic[I][1] for I in range(pdim)]
            if all([it == 0 for it in vt]) and all([it == 1.0 for it in vs]):
                dims = [int(it) for it in self.db.getLogicSize()]
                value = dims[dir] // 2
                return value,[0, int(dims[dir]) - 1, 1]
            else:
                A, B = self.getPhysicBox()[dir]
                value = (A + B) / 2.0
                return value,[A, B, 0]

    def toPhysic(self, value):
        dir = self.direction.value
        pdim = self.getPointDim()
        vt = [self.logic_to_physic[I][0] for I in range(pdim)]
        vs = [self.logic_to_physic[I][1] for I in range(pdim)]
        p1,p2=value
        p1 = [vs[I] * p1[I] + vt[I] for I in range(pdim)]
        p2 = [vs[I] * p2[I] + vt[I] for I in range(pdim)]
        if pdim==1:
            assert(len(p1)==1 and len(p2)==1)
            p1.append(0.0); p2.append(1.0)
        elif pdim==3:
            del p1[dir]; del p2[dir]
        x1,y1=p1; x2,y2=p2
        return [x1,y1, x2-x1, y2-y1]

    def toLogic(self, value):
        pdim = self.getPointDim()
        dir = self.direction.value
        vt = [self.logic_to_physic[I][0] for I in range(pdim)]
        vs = [self.logic_to_physic[I][1] for I in range(pdim)]
        x,y,w,h=value
        p1=[x  ,y  ]
        p2=[x+w,y+h]
        if pdim==1:
            del p1[1]; del p2[1]
        elif pdim==3:
            p1.insert(dir, 0)
            p2.insert(dir, 0)
        p1 = [(p1[I] - vt[I]) / vs[I] for I in range(pdim)]
        p2 = [(p2[I] - vt[I]) / vs[I] for I in range(pdim)]
        if pdim == 3:
            p1[dir] = int((self.offset.value  - vt[dir]) / vs[dir])
            p2[dir] = p1[dir]+1 
        return [p1, p2]

    # ----- Playback -----
    def togglePlay(self):
        if self.play.is_playing:
            self.stopPlay()
        else:
            self.startPlay()

    def startPlay(self):
        logger.info(f"id={self.id}::startPlay")
        self.play.is_playing = True
        self.range_mode.value='user'
        self.play_button.name = "Stop"
        self.play.t1 = time.time()
        self.play.wait_render_id = None
        self.play.num_refinements = self.num_refinements.value
        self.num_refinements.value = 1
        self.setWidgetsDisabled(True)
        self.play_button.disabled = False

    def stopPlay(self):
        logger.info(f"id={self.id}::stopPlay")
        self.play.is_playing = False
        self.view_dependent.value=True
        self.play.wait_render_id = None
        self.num_refinements.value = self.play.num_refinements
        self.setWidgetsDisabled(False)
        self.play_button.disabled = False
        self.play_button.name = "Play"

    def playNextIfNeeded(self):
        if not self.play.is_playing:
            return
        t2 = time.time()
        if (t2 - self.play.t1) < float(self.play_sec.value):
            return
        if self.play.wait_render_id is not None and self.render_id.value<self.play.wait_render_id:
            return
        T = int(self.timestep.value) + self.timestep_delta.value
        if T >= self.timestep.end:
            T = self.timestep.start
        self.play.wait_render_id = self.render_id.value+1
        self.play.t1 = time.time()
        self.timestep.value= T

    def onShowMetadataClick(self):
        self.metadata.visible = not self.metadata.visible

    def setWidgetsDisabled(self, value):
        self.scene.disabled = value
        self.palette.disabled = value
        self.timestep.disabled = value
        self.timestep_delta.disabled = value
        self.field.disabled = value
        self.direction.disabled = value
        self.offset.disabled = value
        self.num_refinements.disabled = value
        self.resolution.disabled = value
        self.view_dependent.disabled = value
        self.request.disabled = value
        self.response.disabled = value
        self.play_button.disabled = value
        self.play_sec.disabled = value

    def getPointDim(self):
        return self.db.getPointDim() if self.db else 2

    # ----- Query orchestration -----
    def refresh(self):
        self.aborted.setTrue()
        self.new_job=True

    def getQueryLogicBox(self):
        viewport=self.canvas.getViewport()
        return self.toLogic(viewport)

    def setQueryLogicBox(self,value):
        viewport=self.toPhysic(value)
        self.canvas.setViewport(viewport)
        self.refresh()

    def getLogicCenter(self):
        pdim=self.getPointDim()  
        p1,p2=self.getQueryLogicBox()
        assert(len(p1)==pdim and len(p2)==pdim)
        return [(p1[I]+p2[I])*0.5 for I in range(pdim)]

    def getLogicSize(self):
        pdim=self.getPointDim()
        p1,p2=self.getQueryLogicBox()
        assert(len(p1)==pdim and len(p2)==pdim)
        return [(p2[I]-p1[I]) for I in range(pdim)]

    def gotNewData(self, result):
        data=result['data']
        try:
            data_range=np.nanmin(data),np.nanmax(data)
        except:
            data_range=0.0,0.0

        logic_box=result['logic_box'] 
        mode=self.range_mode.value

        maxh=self.db.getMaxResolution()
        dir=self.direction.value
        pdim=self.getPointDim()
        vt,vs=self.logic_to_physic[dir] if pdim==3 else (0.0,1.0)
        endh=result['H']

        user_physic_offset=self.offset.value
        real_logic_offset=logic_box[0][dir] if pdim==3 else 0.0
        real_physic_offset=vs*real_logic_offset + vt 
        user_logic_offset=int((user_physic_offset-vt)/vs)

        self.offset.name=" ".join([
            f"Offset: {user_physic_offset:.3f}±{abs(user_physic_offset-real_physic_offset):.3f}",
            f"Pixel: {user_logic_offset}±{abs(user_logic_offset-real_logic_offset)}",
            f"Max Res: {endh}/{maxh}"
        ])

        if mode=="dynamic":
            self.range_min.value = round(data_range[0],6)
            self.range_max.value = round(data_range[1],6)
        if mode=="dynamic-acc":
            if self.range_min.value==self.range_max.value:
                self.range_min.value=data_range[0]
                self.range_max.value=data_range[1]
            else:
                self.range_min.value = min(self.range_min.value, data_range[0])
                self.range_max.value = max(self.range_max.value, data_range[1])
        low = cdouble(self.range_min.value)
        high= cdouble(self.range_max.value)

        if self.color_bar is None:
            color_mapper_type=self.color_mapper_type.value
            is_log=color_mapper_type=="log"
            palette=self.palette.value
            mapper_low =max(EPSILON, low ) if is_log else low
            mapper_high=max(EPSILON, high) if is_log else high
            self.color_bar = bokeh.models.ColorBar(color_mapper = 
                bokeh.models.LogColorMapper   (palette=palette, low=mapper_low, high=mapper_high) if is_log else 
                bokeh.models.LinearColorMapper(palette=palette, low=mapper_low, high=mapper_high)
            )
        data = np.ascontiguousarray(data[::-1, :])
        self.canvas.showData(data, self.toPhysic(logic_box), color_bar=self.color_bar)
        self.stats_view.set_data(data)
        try:
            regional_data=data[int(self.selected_logic_box[0][1]):int(self.selected_logic_box[1][1]),
                               int(self.selected_logic_box[0][0]):int(self.selected_logic_box[1][0])]
            if self.region_stats_panel.visible==True:
                self.region_stats_view.set_data(regional_data)
        except: 
            pass

        (X,Y,Z),(tX,tY,tZ)=self.getLogicAxis()
        self.canvas.setAxisLabels(tX,tY)

        tot_pixels=np.prod(data.shape)
        self.H=result['H']
        query_status="running" if result['running'] else "FINISHED"
        self.response.value=" ".join([
            f"#{result['I']+1}",
            f"{str(logic_box).replace(' ','')}",
            str(data.shape),
            f"Res={result['H']}/{maxh}",
            f"{result['msec']}msec",
            str(query_status)
        ])

        self.render_id.value=self.render_id.value+1 
  
    def pushJobIfNeeded(self):
        if not self.new_job:
            return

        canvas_w,canvas_h=(self.canvas.getWidth(),self.canvas.getHeight())
        query_logic_box=self.getQueryLogicBox()
        pdim=self.getPointDim()

        self.aborted.setTrue()
        self.query_node.waitIdle()
        self.query_node2.waitIdle()
        num_refinements = self.num_refinements.value
        if num_refinements==0:
            num_refinements={1: 1, 2: 3, 3: 4}[pdim]
        self.aborted=Aborted()

        if (time.time()-self.last_job_pushed)<0.2:
            return
		
        if not self.view_dependent.value:
            endh=self.resolution.value
            max_pixels=None
        else:
            endh=None 
            canvas_w,canvas_h=(self.canvas.getWidth(),self.canvas.getHeight())
            if not canvas_w or not canvas_h:
                return
            if pdim==1:
                max_pixels=canvas_w
            else:
                delta=self.resolution.value-self.getMaxResolution()
                a,b=self.resolution.value,self.getMaxResolution()
                if a==b:
                    coeff=1.0
                elif a<b:
                    coeff=1.0/pow(1.3,abs(delta))
                else:
                    coeff=1.0*pow(1.3,abs(delta))
                max_pixels=int(canvas_w*canvas_h*coeff)
			
        self.scene_body.value=json.dumps(self.getSceneBody(),indent=2)
        timestep=int(self.timestep.value)
        field="2T"  # <— the only field

        box_i=[[int(it) for it in jt] for jt in query_logic_box]
        self.request.value=f"t={timestep} b={str(box_i).replace(' ','')} {canvas_w}x{canvas_h}"
        self.response.value="Running..."

        self.query_node.pushJob(
            self.db, 
            access=self.access,
            timestep=timestep, 
            field=field, 
            logic_box=query_logic_box, 
            max_pixels=max_pixels, 
            num_refinements=num_refinements, 
            endh=endh, 
            aborted=self.aborted
        )
		
        self.last_job_pushed=time.time()
        self.new_job=False

    # ----- Selection details dialog -----
    def showDetails(self,evt=None):
        import openvisuspy as ovy
        import panel as pn

        self.region_view.reset_view()
        x,y,w,h=evt.new
        z=int(self.offset.value)
        logic_box=self.toLogic([x,y,w,h])
        self.logic_box=logic_box
        data=list(ovy.ExecuteBoxQuery(self.db, access=self.db.createAccess(), field=self.field.value,
                                      timestep=self.timestep.value,logic_box=logic_box,num_refinements=1))[0]["data"]
        self.selected_logic_box=self.logic_box
        self.selected_physic_box=[[x,x+w],[y,y+h]]
        self.detailed_data=data

        save_numpy_button = pn.widgets.Button(name='Save Data as Numpy', button_type='primary')
        download_script_button = pn.widgets.Button(name='Download Script', button_type='primary')
        apply_colormap_button = pn.widgets.Button(name='Replace Existing Range', button_type='primary')
        apply_min_colormap_button = pn.widgets.Button(name='Replace Min Range', button_type='primary')
        apply_max_colormap_button = pn.widgets.Button(name='Replace Max Range', button_type='primary')
        apply_avg_min_colormap_button = pn.widgets.Button(name='Apply Average Min', button_type='primary')
        apply_avg_max_colormap_button = pn.widgets.Button(name='Apply Average Max', button_type='primary')

        save_numpy_button.on_click(self.save_data)
        download_script_button.on_click(self.download_script)
        apply_colormap_button.on_click(self.apply_cmap)
        apply_max_colormap_button.on_click(self.apply_max_cmap)
        apply_min_colormap_button .on_click(self.apply_min_cmap)
        apply_avg_max_colormap_button.on_click(self.apply_avg_max_cmap)
        apply_avg_min_colormap_button .on_click(self.apply_avg_min_cmap)

        self.vmin,self.vmax=np.nanmin(data),np.nanmax(data)
        add_range_button=pn.widgets.Button(name='Add This Range',button_type='primary')
        add_range_button.on_click(self.add_range)

        if self.range_mode.value=="dynamic-acc":
            self.vmin,self.vmax=np.nanmin(data),np.nanmax(data)
            self.range_min.value = min(self.range_min.value, self.vmin)
            self.range_max.value = max(self.range_max.value, self.vmax)
            logger.info(f"Updating range with selected area vmin={self.vmin} vmax={self.vmax}")

        p = figure(x_range=(self.selected_physic_box[0][0], self.selected_physic_box[0][1]),
                   y_range=(self.selected_physic_box[1][0], self.selected_physic_box[1][1]))
        palette_name = self.palette.value_name if self.palette.value_name.endswith("256") else "Turbo256"
        mapper = LinearColorMapper(palette=palette_name, low=np.nanmin(self.detailed_data), high=np.nanmax(self.detailed_data))
        data_flipped = np.ascontiguousarray(data[::-1, :])
        source = ColumnDataSource(data=dict(image=[data]))
        dw = abs(self.selected_physic_box[0][1] -self.selected_physic_box[0][0])
        dh = abs(self.selected_physic_box[1][1] - self.selected_physic_box[1][0])
        p.image(image='image', x=self.selected_physic_box[0][0], y=self.selected_physic_box[1][0], dw=dw, dh=dh, color_mapper=mapper, source=source)  
        self.region_view.reset_view()
        self.region_view.set_latlon(data, self.selected_physic_box[0][0], self.selected_physic_box[1][0], dw, dh)

        try: self.region_stats_view.set_data(data) 
        except: pass

        color_bar = ColorBar(color_mapper=mapper, label_standoff=12, location=(0,0))
        p.add_layout(color_bar, 'right')
        p.xaxis.axis_label = "Longitude"
        p.yaxis.axis_label = "Latitude"

        self.showDialog(
            pn.Column(
                self.file_name_input, 
                pn.Row(save_numpy_button,download_script_button),
                pn.Row(pn.pane.Bokeh(p), pn.Column(
                    pn.pane.Markdown(f"#### Palette Used: {palette_name}"),
                    pn.pane.Markdown(f"#### New Min/Max Found"),
                    pn.pane.Markdown(f"**Min:** {self.vmin}, **Max:** {self.vmax}"),
                    pn.Row(apply_avg_min_colormap_button,apply_avg_max_colormap_button),
                    add_range_button,
                    apply_colormap_button)),
                sizing_mode="stretch_both"
            ), 
            width=1048, height=748, name="Details"
        )

    # ----- Range apply helpers -----
    def apply_min_cmap(self,event):
        self.range_min.value=self.vmin
        self.range_mode.value="user"
        ShowInfoNotification('New min range applied successfully')

    def add_range(self,event):
        if self.range_max.value<self.vmax:
            self.range_max.value=self.vmax
        if self.range_min.value>self.vmin:
            self.range_min.value=self.vmin
        ShowInfoNotification('Range Added successfully')
     
    def apply_max_cmap(self,event):
        self.range_max.value=self.vmax
        self.range_mode.value="user"
        ShowInfoNotification('New max range applied successfully')
  
    def apply_avg_min_cmap(self,event):
        new_avg_min=(self.range_min.value+self.vmin)/2
        self.range_min.value=round(new_avg_min, 4)
        self.range_mode.value="user"
        ShowInfoNotification('Average Min range applied successfully')

    def apply_avg_max_cmap(self,event):
        new_avg_max=(self.range_max.value+self.vmax)/2
        self.range_max.value=round(new_avg_max, 4)
        self.range_mode.value="user"
        ShowInfoNotification('Average Max range applied successfully')
  
    def apply_cmap(self,event):
        self.range_min.value=self.vmin
        self.range_max.value=self.vmax
        self.range_mode.value="user"
        ShowInfoNotification('New Colormap Range applied successfully')
        self.refresh()

    def download_script(self,event):
        url=self.data_url
        rounded_logic_box = [
            [int(self.logic_box[0][0]), int(self.logic_box[0][1]), self.logic_box[0][2]],
            [int(self.logic_box[1][0]), int(self.logic_box[1][1]), self.logic_box[1][2]]
        ]
        python_file_content = f"""
import OpenVisus
import numpy as np

data_url="{url}"
db=OpenVisus.LoadDataset(data_url)
data=db.read(time={self.timestep.value},logic_box={rounded_logic_box})
np.savez('selected_data',data=data)
"""
        file_path = f'./download_script_{rounded_logic_box[0][0]}_{rounded_logic_box[0][1]}.py'
        with open(file_path, 'w') as file:
            file.write(python_file_content)
        ShowInfoNotification('Script to download selected data saved!')
        print("Script saved successfully.") 

    def save_data(self, event):
        if self.detailed_data is not None:
            file_name = f"{self.file_name_input.value}.npz" if self.file_name_input.value else "test_region.npz"
            np.savez(file_name, data=self.detailed_data, lon_lat=self.selected_physic_box)
            ShowInfoNotification('Data Saved successfully to current directory!')
        else:
            print("No data to save.")

    # ----- Idle loop -----
    def onIdle(self):
        if not self.db:
            return

        self.canvas.onIdle()
        if self.canvas and self.canvas.getWidth()>0 and self.canvas.getHeight()>0:
            self.playNextIfNeeded()

        if self.query_node:
            result=self.query_node.popResult(last_only=True) 
            if result is not None: 
                self.gotNewData(result)
                self.stats_view.set_data(result['data'])
                try:
                    regional_data=result['data'][int(self.selected_logic_box[0][1]):int(self.selected_logic_box[1][1]),
                                                 int(self.selected_logic_box[0][0]):int(self.selected_logic_box[1][0])]
                    if self.region_stats_panel.visible==True:
                        self.region_stats_view.set_data(regional_data)
                except: 
                    pass
            self.pushJobIfNeeded()

