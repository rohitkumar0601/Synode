"use client";

import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

/** Plotly bound to the minified dist bundle. Only ever loaded through a dynamic import (ssr: false). */
const PlotlyChart = createPlotlyComponent(Plotly);
export default PlotlyChart;
