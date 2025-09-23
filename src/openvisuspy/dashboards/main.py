import os
import sys
import logging
import base64
import json
import panel as pn

#sys.path.append('/Users/aashish/Research/github/test/openvpy/src')
#sys.path.append('/glade/work/dpanta/github/openvpy/src')
#sys.path.append('/glade/campaign/work/dpanta/github/openv-ncar-dashboard/src')
sys.path.append('/glade/u/home/dpanta/github/openv-ncar-dashboard/src')
from openvisuspy import SetupLogger, Slice, ProbeTool, GetQueryParams

class DashboardApp:
    def __init__(self, config):
        self.slice = Slice()
        self.slice.load(config)
        self.setup_logging()
        self.setup_layout()

    def setup_logging(self):
        log_filename = os.environ.get("OPENVISUSPY_DASHBOARDS_LOG_FILENAME", "/tmp/openvisuspy-dashboards.log")
        self.logger = SetupLogger(log_filename=log_filename, logging_level=logging.DEBUG)

    def setup_layout(self):
        query_params = GetQueryParams()
        if "load" in query_params:
            body = json.loads(base64.b64decode(query_params['load']).decode("utf-8"))
            self.slice.setSceneBody(body)
        elif "dataset" in query_params:
            scene_name = query_params["dataset"]
            self.slice.scene.value = scene_name
        title = pn.pane.HTML(
            """
            <div style="width: 100%; padding: 0px; display: flex; justify-content: space-between; 
                        align-items: center; box-sizing: border-box;">
                <div style="flex-grow: 1;">
                    <h2 style="margin: 0; font-size: 1.2em; line-height: 1.1;">
                        ERA5 Interactive Dashboard
                    </h2>
                </div>
                <div style="text-align: right; font-size: 11px; line-height: 1.1;">
                    Powered by <a href="https://www.gdex.ucar.edu" target="_blank">NSF NCAR</a> and
                    <a href="https://nationalsciencedatafabric.org/" target="_blank">National Science Data Fabric (NSDF)</a>
                </div>
            </div>
            """,
            sizing_mode="stretch_width"
        )


        # Choose main layout
        if False:
            main_layout = ProbeTool(self.slice).getMainLayout()
        else:
            main_layout = self.slice.getMainLayout()

        # Combine title and layout
        self.app = pn.Column(
            title,
            pn.layout.Divider(),
            main_layout,
            sizing_mode="stretch_width"
        )

    def servable(self):
        return self.app

if __name__.startswith('bokeh'):
    pn.extension(
        "ipywidgets",
        "floatpanel",
        # log_level="DEBUG",
        notifications=True,
        sizing_mode="stretch_width"
    )
    config = sys.argv[1] 
    app_instance = DashboardApp(config)
    app_instance.servable().servable()
