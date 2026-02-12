import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  getCorrelationHeatmap,
  getCorrelationMatrix,
  analyzeCorrelation,
  getCurrentDataset,
} from "@/lib/api";
import ChartContainer from "@/components/charts/ChartContainer";
import { Loader2, AlertCircle } from "lucide-react";

const METHODS = ["pearson", "spearman", "kendall"] as const;

export default function RelationshipsTab() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["correlationHeatmap"],
    queryFn: getCorrelationHeatmap,
  });

  const datasetQuery = useQuery({
    queryKey: ["currentDataset"],
    queryFn: getCurrentDataset,
    staleTime: 15_000,
  });

  const matrixQuery = useQuery({
    queryKey: ["correlationMatrix", datasetQuery.data?.filtered_rows],
    queryFn: getCorrelationMatrix,
    enabled: Boolean(datasetQuery.data?.numeric_columns?.length),
  });

  const [col1, setCol1] = useState<string>("");
  const [col2, setCol2] = useState<string>("");
  const [method, setMethod] = useState<(typeof METHODS)[number]>("pearson");

  useEffect(() => {
    if (!datasetQuery.data?.numeric_columns?.length) return;
    if (!col1) {
      setCol1(datasetQuery.data.numeric_columns[0]);
    }
    if (!col2 && datasetQuery.data.numeric_columns.length > 1) {
      setCol2(datasetQuery.data.numeric_columns[1]);
    }
  }, [datasetQuery.data?.numeric_columns, col1, col2]);

  const correlationMutation = useMutation({
    mutationFn: () =>
      analyzeCorrelation({ col1, col2, method }),
  });

  const strongPairs = useMemo(() => {
    if (!matrixQuery.data) return [];
    const results: { pair: string; value: number }[] = [];
    matrixQuery.data.columns.forEach((rowCol, i) => {
      matrixQuery.data.columns.forEach((col, j) => {
        if (j <= i) return;
        const value = Math.abs(matrixQuery.data.matrix[i][j]);
        if (!Number.isFinite(value)) return;
        if (value >= 0.4) {
          results.push({ pair: `${rowCol} ↔ ${col}`, value });
        }
      });
    });
    return results.sort((a, b) => b.value - a.value).slice(0, 3);
  }, [matrixQuery.data]);

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
        <span>Not enough numeric columns for correlation analysis</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <ChartContainer
        title="Correlation Heatmap"
        description="Hover to inspect co-movements across every numeric field"
        figure={data.figure}
      />

      <section className="rounded-lg border border-navy-500 bg-navy-700 p-4">
        <h4 className="mb-3 text-sm font-semibold uppercase tracking-wider text-orange-500">
          Pairwise Analysis
        </h4>
        <div className="grid gap-3 md:grid-cols-4">
          <div className="space-y-1">
            <label className="text-[10px] uppercase text-gray-400">Method</label>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value as typeof method)}
              className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
            >
              {METHODS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-[10px] uppercase text-gray-400">Variable A</label>
            <select
              value={col1}
              onChange={(e) => setCol1(e.target.value)}
              className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
            >
              {datasetQuery.data?.numeric_columns?.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-[10px] uppercase text-gray-400">Variable B</label>
            <select
              value={col2}
              onChange={(e) => setCol2(e.target.value)}
              className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
            >
              {datasetQuery.data?.numeric_columns?.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-end">
            <button
              className="w-full rounded bg-orange-500 py-2 text-sm font-semibold text-black disabled:cursor-not-allowed disabled:bg-orange-500/40"
              onClick={() => correlationMutation.mutate()}
              disabled={!col1 || !col2 || col1 === col2 || correlationMutation.isPending}
            >
              {correlationMutation.isPending ? "Analyzing..." : "Run Correlation"}
            </button>
          </div>
        </div>

        {correlationMutation.data && (
          <div className="mt-4 grid gap-4 rounded border border-navy-500 bg-navy-600 p-4 text-sm">
            <p>
              <span className="text-gray-400">r</span> = {correlationMutation.data.correlation.toFixed(3)}
              {" • "}
              p-value {correlationMutation.data.p_value.toFixed(4)}
              {" • "}
              Strength {correlationMutation.data.strength}
            </p>
            <p className="text-gray-300">
              {correlationMutation.data.significant
                ? "Relationship is statistically significant."
                : "No statistically significant relationship detected."}
            </p>
          </div>
        )}
      </section>

      <section className="rounded-lg border border-navy-500 bg-navy-700 p-4">
        <h4 className="mb-2 text-sm font-semibold uppercase tracking-wider text-orange-500">
          Strongest Signals
        </h4>
        {matrixQuery.isLoading && <Loader2 className="h-4 w-4 animate-spin text-orange-500" />}
        {!matrixQuery.isLoading && strongPairs.length === 0 && (
          <p className="text-xs text-gray-400">No pairs exceeded |0.4| correlation.</p>
        )}
        {!matrixQuery.isLoading && strongPairs.length > 0 && (
          <ul className="space-y-1 text-sm text-gray-200">
            {strongPairs.map((item) => (
              <li key={item.pair}>
                {item.pair}
                <span className="ml-2 text-gray-400">r = {item.value.toFixed(3)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
