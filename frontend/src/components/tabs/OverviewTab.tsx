import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  getDatasetOverview,
  getCurrentDataset,
  getDistributionPlot,
  getCountPlot,
  getScatterPlot,
} from "@/lib/api";
import { Loader2, AlertCircle } from "lucide-react";
import ChartContainer from "@/components/charts/ChartContainer";

export default function OverviewTab() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["overview"],
    queryFn: getDatasetOverview,
  });

  const datasetQuery = useQuery({
    queryKey: ["currentDataset"],
    queryFn: getCurrentDataset,
    staleTime: 15_000,
  });

  const numericCol = datasetQuery.data?.numeric_columns?.[0];
  const categoricalCol = datasetQuery.data?.categorical_columns?.[0];
  const scatterPair = useMemo(() => {
    const cols = datasetQuery.data?.numeric_columns ?? [];
    if (cols.length < 2) return null;
    return cols.slice(0, 2);
  }, [datasetQuery.data?.numeric_columns]);

  const histogramQuery = useQuery({
    queryKey: ["overview", "hist", numericCol, datasetQuery.data?.filtered_rows],
    queryFn: () => getDistributionPlot(numericCol!),
    enabled: Boolean(numericCol),
  });

  const barQuery = useQuery({
    queryKey: ["overview", "count", categoricalCol, datasetQuery.data?.filtered_rows],
    queryFn: () => getCountPlot(categoricalCol!),
    enabled: Boolean(categoricalCol),
  });

  const scatterQuery = useQuery({
    queryKey: ["overview", "scatter", scatterPair, datasetQuery.data?.filtered_rows],
    queryFn: () =>
      getScatterPlot(scatterPair![0], scatterPair![1], datasetQuery.data?.categorical_columns?.[0]),
    enabled: Boolean(scatterPair),
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-orange-500" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center gap-2 py-10 text-red-400">
        <AlertCircle className="h-5 w-5" />
        <span>Failed to load overview</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Metric cards */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Rows" value={data.row_count.toLocaleString()} />
        <MetricCard label="Columns" value={String(data.column_count)} />
        <MetricCard
          label="Completeness"
          value={`${(data.completeness_score * 100).toFixed(1)}%`}
        />
      </div>

      {/* Data health summary */}
      <section className="rounded-lg border border-navy-500 bg-navy-700 p-4">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-orange-500">
          Data Health
        </h3>
        <p className="text-sm text-gray-300 leading-relaxed">{data.summary}</p>
      </section>

      {/* Describe */}
      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-orange-500">
          df.describe()
        </h3>
        <DescribeTable rows={data.describe} />
      </section>

      {/* Insights */}
      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-orange-500">
          Insights
        </h3>
        <ul className="space-y-1.5">
          {data.insights.map((text, i) => (
            <li key={i} className="text-sm text-gray-300">
              &bull; {text}
            </li>
          ))}
        </ul>
      </section>

      {/* Adaptive charts */}
      <section className="grid gap-4 lg:grid-cols-2">
        {numericCol && (
          <ChartContainer
            title={`Distribution of ${numericCol}`}
            description="Interactive histogram with drill-down"
            figure={histogramQuery.data?.figure}
            isLoading={histogramQuery.isLoading}
            error={histogramQuery.isError ? "Unable to render histogram" : undefined}
          />
        )}

        {categoricalCol && (
          <ChartContainer
            title={`${categoricalCol} Breakdown`}
            description="Bar chart that adapts to filtered selections"
            figure={barQuery.data?.figure}
            isLoading={barQuery.isLoading}
            error={barQuery.isError ? "Unable to render bar chart" : undefined}
          />
        )}

        {scatterPair && (
          <ChartContainer
            title={`${scatterPair[0]} vs ${scatterPair[1]}`}
            description="Scatterplot with optional categorical coloring"
            figure={scatterQuery.data?.figure}
            isLoading={scatterQuery.isLoading}
            error={scatterQuery.isError ? "Unable to render scatterplot" : undefined}
            className="lg:col-span-2"
          />
        )}
      </section>

      {/* Data preview */}
      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-orange-500">
          Data Preview
        </h3>
        <div className="overflow-x-auto rounded border border-navy-500">
          <table className="min-w-full text-sm">
            <thead className="bg-navy-700 text-xs uppercase text-gray-400">
              <tr>
                {data.preview.length > 0 &&
                  Object.keys(data.preview[0]).map((col) => (
                    <th key={col} className="px-3 py-2 text-left font-medium">
                      {col}
                    </th>
                  ))}
              </tr>
            </thead>
            <tbody>
              {data.preview.map((row, i) => (
                <tr
                  key={i}
                  className="border-t border-navy-500 hover:bg-navy-700/50"
                >
                  {Object.values(row).map((val, j) => (
                    <td key={j} className="whitespace-nowrap px-3 py-1.5 text-gray-300">
                      {val == null ? "—" : String(val)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

interface DescribeTableProps {
  rows: { column: string; stats: Record<string, string | number | null> }[];
}

function DescribeTable({ rows }: DescribeTableProps) {
  if (!rows.length) {
    return <p className="text-xs text-gray-400">Describe summary unavailable.</p>;
  }

  const statKeys = Array.from(
    new Set(
      rows.flatMap((row) => Object.keys(row.stats ?? {}))
    )
  );

  return (
    <div className="overflow-auto rounded border border-navy-500">
      <table className="min-w-full text-xs">
        <thead className="bg-navy-700 text-[11px] uppercase text-gray-400">
          <tr>
            <th className="px-3 py-2 text-left">Column</th>
            {statKeys.map((key) => (
              <th key={key} className="px-3 py-2 text-left font-medium">
                {key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.column} className="border-t border-navy-500">
              <td className="px-3 py-2 font-medium text-gray-200">{row.column}</td>
              {statKeys.map((key) => (
                <td key={key} className="px-3 py-2 text-gray-300">
                  {formatStatValue(row.stats[key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatStatValue(value: string | number | null | undefined) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toFixed(3);
  }
  return value;
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-navy-500 bg-navy-700 p-4">
      <p className="text-xs font-medium uppercase text-gray-400">{label}</p>
      <p className="mt-1 text-2xl font-bold text-white">{value}</p>
    </div>
  );
}
