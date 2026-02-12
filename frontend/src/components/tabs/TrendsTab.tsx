import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  getCurrentDataset,
  analyzeTrend,
  getTimeSeriesFigure,
} from "@/lib/api";
import ChartContainer from "@/components/charts/ChartContainer";
import { Loader2 } from "lucide-react";

const CHART_TYPES: Array<{ key: "line" | "area" | "stacked"; label: string }> = [
  { key: "line", label: "Line" },
  { key: "area", label: "Area" },
  { key: "stacked", label: "Stacked" },
];

const ROLLING_WINDOWS = [0, 3, 5, 12];

export default function TrendsTab() {
  const datasetQuery = useQuery({
    queryKey: ["currentDataset"],
    queryFn: getCurrentDataset,
    staleTime: 15_000,
  });

  const numericCols = datasetQuery.data?.numeric_columns ?? [];
  const dateLike = useMemo(() => {
    const cols = datasetQuery.data?.column_names ?? [];
    return cols.filter((name) => /date|time|year/i.test(name));
  }, [datasetQuery.data?.column_names]);

  const [xAxis, setXAxis] = useState<string>("");
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [chartType, setChartType] = useState<"line" | "area" | "stacked">("line");
  const [rollingWindow, setRollingWindow] = useState<number>(0);

  useEffect(() => {
    if (!selectedMetrics.length && numericCols.length) {
      setSelectedMetrics(numericCols.slice(0, Math.min(2, numericCols.length)));
    }
  }, [numericCols, selectedMetrics.length]);

  useEffect(() => {
    if (!xAxis && dateLike.length) {
      setXAxis(dateLike[0]);
    }
  }, [dateLike, xAxis]);

  const figureQuery = useQuery({
    queryKey: [
      "trendFigure",
      xAxis || "sequence",
      selectedMetrics,
      chartType,
      rollingWindow,
      datasetQuery.data?.filtered_rows,
    ],
    queryFn: () =>
      getTimeSeriesFigure({
        x_column: xAxis || undefined,
        value_columns: selectedMetrics,
        chart_type: chartType,
        rolling_window: rollingWindow || undefined,
      }),
    enabled: selectedMetrics.length > 0,
  });

  const summaryQuery = useQuery({
    queryKey: ["trendSummary", selectedMetrics, datasetQuery.data?.filtered_rows],
    queryFn: async () => Promise.all(selectedMetrics.map((col) => analyzeTrend({ column: col }))),
    enabled: selectedMetrics.length > 0,
  });

  return (
    <div className="space-y-6">
      <ChartContainer
        title="Trends Explorer"
        description="Overlay multiple metrics, switch chart styles, and smooth with rolling windows"
        figure={figureQuery.data?.figure}
        isLoading={figureQuery.isLoading}
        error={
          figureQuery.isError
            ? "Unable to render trend chart"
            : !figureQuery.isLoading && selectedMetrics.length === 0
              ? "Select at least one metric"
              : undefined
        }
      />

      <section className="grid gap-4 rounded-lg border border-navy-500 bg-navy-700 p-4 lg:grid-cols-3">
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-orange-500">Metrics</p>
          <div className="max-h-40 space-y-1 overflow-y-auto">
            {numericCols.map((col) => (
              <label key={col} className="flex items-center gap-2 text-sm text-gray-200">
                <input
                  type="checkbox"
                  checked={selectedMetrics.includes(col)}
                  onChange={() => toggleMetric(col, selectedMetrics, setSelectedMetrics)}
                />
                {col}
              </label>
            ))}
            {numericCols.length === 0 && (
              <p className="text-xs text-gray-400">No numeric columns detected.</p>
            )}
          </div>
        </div>

        <div className="space-y-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-orange-500">Chart Type</p>
            <div className="mt-2 flex gap-2">
              {CHART_TYPES.map((type) => (
                <button
                  key={type.key}
                  className={`flex-1 rounded px-2 py-1 text-sm ${chartType === type.key ? "bg-orange-500 text-black" : "bg-navy-600 text-gray-300"
                    }`}
                  onClick={() => setChartType(type.key)}
                >
                  {type.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-orange-500">Rolling Window</p>
            <div className="mt-2 grid grid-cols-4 gap-2">
              {ROLLING_WINDOWS.map((window) => (
                <button
                  key={window}
                  className={`rounded px-2 py-1 text-sm ${rollingWindow === window ? "bg-orange-500 text-black" : "bg-navy-600 text-gray-300"
                    }`}
                  onClick={() => setRollingWindow(window)}
                >
                  {window === 0 ? "None" : `${window}`}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-orange-500">
            X-Axis
          </label>
          <select
            value={xAxis}
            onChange={(e) => setXAxis(e.target.value)}
            className="w-full rounded border border-navy-500 bg-navy-600 px-3 py-2 text-sm focus:border-orange-500 focus:outline-none"
          >
            <option value="">Row Order</option>
            {dateLike.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>
      </section>

      <section className="space-y-3 rounded-lg border border-navy-500 bg-navy-700 p-4">
        <h4 className="text-sm font-semibold uppercase tracking-wider text-orange-500">
          Trend Narratives
        </h4>
        {summaryQuery.isLoading && <Loader2 className="h-4 w-4 animate-spin text-orange-500" />}
        {!summaryQuery.isLoading && (!summaryQuery.data || summaryQuery.data.length === 0) && (
          <p className="text-xs text-gray-400">Select one or more metrics to see automated analysis.</p>
        )}
        {summaryQuery.data && summaryQuery.data.length > 0 && (
          <ul className="space-y-2 text-sm text-gray-200">
            {summaryQuery.data.map((result, idx) => (
              <li key={`${result.column}-${idx}`} className="rounded border border-navy-600 bg-navy-600/60 p-3">
                <p className="font-semibold">{result.column}</p>
                <p className="text-gray-300">
                  {describeTrend(result)}
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function toggleMetric(
  metric: string,
  selected: string[],
  setter: (value: string[]) => void
) {
  if (selected.includes(metric)) {
    setter(selected.filter((m) => m !== metric));
  } else {
    setter([...selected, metric]);
  }
}

function describeTrend(result: Awaited<ReturnType<typeof analyzeTrend>>) {
  if (!result) return "Trend unavailable";
  const direction =
    result.trend === "increasing"
      ? "is trending upward"
      : result.trend === "decreasing"
        ? "is trending downward"
        : "is stable";
  const slopeText = result.slope != null ? `slope ${result.slope.toFixed(4)}` : "no slope";
  const r2Text = result.r_squared != null ? `R² ${result.r_squared.toFixed(3)}` : "R² —";
  return `${direction} (${slopeText}, ${r2Text}) with p-value ${result.p_value?.toFixed(4) ?? "—"}.`;
}
