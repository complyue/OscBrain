"""
Common Bokeh Stuff

"""

__all__ = [
    "ColumnDataSource",
    "figure",
    "show",
    "push_notebook",
    "Range1d",
    "LinearAxis",
    "HoverTool",
    "CustomJS",
    "Document",
    "column",
    "row",
    "gridplot",
    "DataTable",
    "TableColumn",
    "StringFormatter",
    "DateFormatter",
    "NumberFormatter",
    "BooleanFormatter",
    "HTMLTemplateFormatter",
    "Select",
    "Slider",
    "DateSlider",
    "DateRangeSlider",
    "CheckboxButtonGroup",
    "RadioButtonGroup",
    "RangeSlider",
    "Button",
    "Dropdown",
    "Toggle",
    "TextInput",
    "Div",
    "Paragraph",
    "PreText",
    "BasicTicker",
    "FixedTicker",
    "CustomJSTickFormatter",
    "Span",
    "Whisker",
    "Label",
    "Band",
    "Arrow",
    "value",
    "field",
    "setup_notebook",
]

from bokeh.core.properties import *
from bokeh.embed import *
from bokeh.layouts import *
from bokeh.models import *
from bokeh.plotting import *
from bokeh.io import *

import pandas as pd


def setup_notebook():
    from bokeh.resources import INLINE, Resources
    from bokeh.plotting import output_notebook

    from IPython.display import HTML, Javascript, clear_output, display

    # formatting options
    pd.set_option("expand_frame_repr", False)
    pd.set_option("display.notebook_repr_html", True)
    pd.set_option("display.max_columns", 300)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 2000)
    try:
        pd.set_option("display.precision", 7)
    except:
        pd.set_option("precision", 7)

    output_notebook(resources=INLINE)

    display(
        HTML(
            """
<style>
.container { width:98% !important; }
</style>
"""
        )
    )
