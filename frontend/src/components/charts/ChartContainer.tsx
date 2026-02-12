import { useMemo, useState } from "react";
import PlotlyChart from "./PlotlyChart";
import type { PlotlyFigure } from "@/types/dataset";
import { Table, LineChart } from "lucide-react";

interface Props {
    title: string;
    description?: string;
    figure?: PlotlyFigure["figure"];
    isLoading?: boolean;
    error?: string;
    actions?: React.ReactNode;
    className?: string;
}

export default function ChartContainer({
    title,
    description,
    figure,
    isLoading,
    error,
    actions,
    className,
}: Props) {
    const [mode, setMode] = useState<"chart" | "table">("chart");
    const tableData = useMemo(() => (figure ? extractTableData(figure) : null), [figure]);

    return (
        <section className={`space-y-3 rounded-lg border border-navy-500 bg-navy-700 p-4 ${className ?? ""}`}>
            <header className="flex items-start justify-between gap-4">
                <div>
                    <h3 className="text-sm font-semibold uppercase tracking-wider text-orange-500">
                        {title}
                    </h3>
                    {description && <p className="text-xs text-gray-400">{description}</p>}
                </div>
                <div className="flex items-center gap-2 text-xs">
                    <button
                        className={`flex items-center gap-1 rounded px-2 py-1 ${mode === "chart" ? "bg-orange-500 text-black" : "bg-navy-600 text-gray-300"}`}
                        onClick={() => setMode("chart")}
                    >
                        <LineChart className="h-3.5 w-3.5" /> Chart
                    </button>
                    <button
                        className={`flex items-center gap-1 rounded px-2 py-1 ${mode === "table" ? "bg-orange-500 text-black" : "bg-navy-600 text-gray-300"}`}
                        onClick={() => setMode("table")}
                        disabled={!tableData}
                    >
                        <Table className="h-3.5 w-3.5" /> Table
                    </button>
                    {actions}
                </div>
            </header>

            {isLoading && <p className="text-sm text-gray-400">Loading...</p>}
            {error && <p className="text-sm text-red-400">{error}</p>}

            {!isLoading && !error && figure && mode === "chart" && (
                <PlotlyChart figure={figure} className="h-[360px]" />
            )}

            {!isLoading && !error && mode === "table" && tableData && (
                <div className="max-h-[360px] overflow-auto rounded border border-navy-500">
                    <table className="min-w-full text-xs">
                        <thead className="bg-navy-600 text-left text-[11px] uppercase text-gray-400">
                            <tr>
                                {tableData.columns.map((col) => (
                                    <th key={col} className="px-3 py-2 font-medium">
                                        {col}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {tableData.rows.map((row, idx) => (
                                <tr key={idx} className="border-t border-navy-600 hover:bg-navy-600/40">
                                    {tableData.columns.map((col) => (
                                        <td key={col} className="px-3 py-1 text-gray-200">
                                            {row[col] ?? "—"}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </section>
    );
}

function extractTableData(figure: PlotlyFigure["figure"]) {
    const columns = new Set<string>(["series"]);
    const rows: Record<string, unknown>[] = [];

    for (const [idx, trace] of (figure.data as Plotly.Data[]).entries()) {
        const name = trace.name ?? `Series ${idx + 1}`;
        const type = (trace.type ?? "scatter") as string;

        if (type === "heatmap" && Array.isArray(trace.z) && Array.isArray(trace.x) && Array.isArray(trace.y)) {
            for (let i = 0; i < trace.y.length; i += 1) {
                for (let j = 0; j < trace.x.length; j += 1) {
                    const value = Array.isArray(trace.z[i]) ? trace.z[i][j] : undefined;
                    rows.push({ series: name, x: trace.x[j], y: trace.y[i], value });
                }
            }
            columns.add("x");
            columns.add("y");
            columns.add("value");
            continue;
        }

        const xValues = Array.isArray(trace.x) ? trace.x : [];
        const yValues = Array.isArray(trace.y) ? trace.y : [];
        const limit = Math.max(xValues.length, yValues.length);

        for (let i = 0; i < limit && rows.length < 500; i += 1) {
            rows.push({ series: name, x: xValues[i], y: yValues[i] });
        }
        columns.add("x");
        columns.add("y");
    }

    if (rows.length === 0) {
        return null;
    }

    return { columns: Array.from(columns), rows };
}
