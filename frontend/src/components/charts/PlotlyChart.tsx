import Plot from "react-plotly.js";
import type { PlotlyFigure } from "@/types/dataset";

interface Props {
  figure: PlotlyFigure["figure"];
  className?: string;
}

export default function PlotlyChart({ figure, className }: Props) {
  const layout = {
    ...figure.layout,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#ffffff" },
    xaxis: {
      ...(figure.layout.xaxis as Record<string, unknown> | undefined),
      gridcolor: "#2a3050",
      zerolinecolor: "#2a3050",
    },
    yaxis: {
      ...(figure.layout.yaxis as Record<string, unknown> | undefined),
      gridcolor: "#2a3050",
      zerolinecolor: "#2a3050",
    },
    autosize: true,
    margin: { t: 40, r: 20, b: 60, l: 60 },
  };

  return (
    <div className={className}>
      <Plot
        data={figure.data as Plotly.Data[]}
        layout={layout as Partial<Plotly.Layout>}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        config={{ responsive: true, displayModeBar: false }}
      />
    </div>
  );
}
