from flask import Flask, render_template

# for iris
from bokeh.plotting import figure
from bokeh.sampledata.iris import flowers
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

# for curves
from numpy import linspace
from scipy.stats.kde import gaussian_kde
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
from bokeh.sampledata.perceptions import probly
import colorcet as cc

app = Flask(__name__)


@app.route('/')
def home():
    colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    colors = [colormap[x] for x in flowers['species']]

    p = figure(title="Iris Morphology")
    p.xaxis.axis_label = 'Petal Length'
    p.yaxis.axis_label = 'Petal Width'

    p.circle(
        flowers["petal_length"],
        flowers["petal_width"],
        color=colors,
        fill_alpha=0.2,
        size=10)

    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # render template
    script, div = components(p)
    html = render_template(
        'index.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    )
    return encode_utf8(html)
    return render_template('index.html')


@app.route('/echarts')
def echarts():

    colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    colors = [colormap[x] for x in flowers['species']]

    p = figure(title="Iris Morphology")
    p.xaxis.axis_label = 'Petal Length'
    p.yaxis.axis_label = 'Petal Width'

    p.circle(
        flowers["petal_length"],
        flowers["petal_width"],
        color=colors,
        fill_alpha=0.2,
        size=10)

    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # render template
    script, div = components(p)
    html = render_template(
        'echarts.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    )
    return encode_utf8(html)


@app.route('/iris')
def iris():

    colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    colors = [colormap[x] for x in flowers['species']]

    p = figure(title="Iris Morphology")
    p.xaxis.axis_label = 'Petal Length'
    p.yaxis.axis_label = 'Petal Width'

    p.circle(
        flowers["petal_length"],
        flowers["petal_width"],
        color=colors,
        fill_alpha=0.2,
        size=10)

    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # render template
    script, div = components(p)
    html = render_template(
        'index.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    )
    return encode_utf8(html)


@app.route('/curves')
def curves():
    def joy(category, data, scale=20):
        return list(zip([category] * len(data), scale * data))

    cats = list(reversed(probly.keys()))

    palette = [cc.rainbow[i * 15] for i in range(17)]

    x = linspace(-20, 110, 500)

    source = ColumnDataSource(data=dict(x=x))

    p = figure(
        y_range=cats, plot_width=900, x_range=(-5, 105), toolbar_location=None)

    for i, cat in enumerate(reversed(cats)):
        pdf = gaussian_kde(probly[cat])
        y = joy(cat, pdf(x))
        source.add(y, cat)
        p.patch(
            'x',
            cat,
            color=palette[i],
            alpha=0.6,
            line_color="black",
            source=source)

    p.outline_line_color = None
    p.background_fill_color = "#efefef"

    p.xaxis.ticker = FixedTicker(ticks=list(range(0, 101, 10)))
    p.xaxis.formatter = PrintfTickFormatter(format="%d%%")

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "#dddddd"
    p.xgrid.ticker = p.xaxis[0].ticker

    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None

    p.y_range.range_padding = 0.12

    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # render template
    script, div = components(p)
    html = render_template(
        'index.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    )
    return encode_utf8(html)


''' Present a scatter plot with linked histograms on both axes.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve selection_histogram.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/selection_histogram
in your browser.
'''

import numpy as np

from bokeh.layouts import row, column
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer
from bokeh.plotting import figure, curdoc

# create three normal population samples with different parameters
x1 = np.random.normal(loc=5.0, size=400) * 100
y1 = np.random.normal(loc=10.0, size=400) * 10

x2 = np.random.normal(loc=5.0, size=800) * 50
y2 = np.random.normal(loc=5.0, size=800) * 10

x3 = np.random.normal(loc=55.0, size=200) * 10
y3 = np.random.normal(loc=4.0, size=200) * 10

x = np.concatenate((x1, x2, x3))
y = np.concatenate((y1, y2, y3))

TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset"

# create the scatter plot
p = figure(
    tools=TOOLS,
    plot_width=600,
    plot_height=600,
    min_border=10,
    min_border_left=50,
    toolbar_location="above",
    x_axis_location=None,
    y_axis_location=None,
    title="Linked Histograms")
p.background_fill_color = "#fafafa"
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False

r = p.scatter(x, y, size=3, color="#3A5785", alpha=0.6)

# create the horizontal histogram
hhist, hedges = np.histogram(x, bins=20)
hzeros = np.zeros(len(hedges) - 1)
hmax = max(hhist) * 1.1

LINE_ARGS = dict(color="#3A5785", line_color=None)

ph = figure(
    toolbar_location=None,
    plot_width=p.plot_width,
    plot_height=200,
    x_range=p.x_range,
    y_range=(-hmax, hmax),
    min_border=10,
    min_border_left=50,
    y_axis_location="right")
ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi / 4
ph.background_fill_color = "#fafafa"

ph.quad(
    bottom=0,
    left=hedges[:-1],
    right=hedges[1:],
    top=hhist,
    color="white",
    line_color="#3A5785")
hh1 = ph.quad(
    bottom=0,
    left=hedges[:-1],
    right=hedges[1:],
    top=hzeros,
    alpha=0.5,
    **LINE_ARGS)
hh2 = ph.quad(
    bottom=0,
    left=hedges[:-1],
    right=hedges[1:],
    top=hzeros,
    alpha=0.1,
    **LINE_ARGS)

# create the vertical histogram
vhist, vedges = np.histogram(y, bins=20)
vzeros = np.zeros(len(vedges) - 1)
vmax = max(vhist) * 1.1

pv = figure(
    toolbar_location=None,
    plot_width=200,
    plot_height=p.plot_height,
    x_range=(-vmax, vmax),
    y_range=p.y_range,
    min_border=10,
    y_axis_location="right")
pv.ygrid.grid_line_color = None
pv.xaxis.major_label_orientation = np.pi / 4
pv.background_fill_color = "#fafafa"

pv.quad(
    left=0,
    bottom=vedges[:-1],
    top=vedges[1:],
    right=vhist,
    color="white",
    line_color="#3A5785")
vh1 = pv.quad(
    left=0,
    bottom=vedges[:-1],
    top=vedges[1:],
    right=vzeros,
    alpha=0.5,
    **LINE_ARGS)
vh2 = pv.quad(
    left=0,
    bottom=vedges[:-1],
    top=vedges[1:],
    right=vzeros,
    alpha=0.1,
    **LINE_ARGS)

layout = column(row(p, pv), row(ph, Spacer(width=200, height=200)))

curdoc().add_root(layout)
curdoc().title = "Selection Histogram"


def update(attr, old, new):
    inds = np.array(new['1d']['indices'])
    if len(inds) == 0 or len(inds) == len(x):
        hhist1, hhist2 = hzeros, hzeros
        vhist1, vhist2 = vzeros, vzeros
    else:
        neg_inds = np.ones_like(x, dtype=np.bool)
        neg_inds[inds] = False
        hhist1, _ = np.histogram(x[inds], bins=hedges)
        vhist1, _ = np.histogram(y[inds], bins=vedges)
        hhist2, _ = np.histogram(x[neg_inds], bins=hedges)
        vhist2, _ = np.histogram(y[neg_inds], bins=vedges)

    hh1.data_source.data["top"] = hhist1
    hh2.data_source.data["top"] = -hhist2
    vh1.data_source.data["right"] = vhist1
    vh2.data_source.data["right"] = -vhist2


r.data_source.on_change('selected', update)

if __name__ == '__main__':
    app.run(debug=True)
