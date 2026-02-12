import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getCurrentDataset, getGeospatialMap } from "@/lib/api";
import ChartContainer from "@/components/charts/ChartContainer";
import { AlertCircle, Loader2 } from "lucide-react";

export default function GeospatialTab() {
    const datasetQuery = useQuery({
        queryKey: ["currentDataset"],
        queryFn: getCurrentDataset,
        staleTime: 15_000,
    });

    const [latColumn, setLatColumn] = useState<string>("");
    const [lonColumn, setLonColumn] = useState<string>("");
    const [colorColumn, setColorColumn] = useState<string>("");
    const [latTouched, setLatTouched] = useState(false);
    const [lonTouched, setLonTouched] = useState(false);

    const geoQuery = useQuery({
        queryKey: ["geospatial", latColumn, lonColumn, colorColumn, datasetQuery.data?.filtered_rows],
        queryFn: () =>
            getGeospatialMap({
                lat_column: latColumn || undefined,
                lon_column: lonColumn || undefined,
                color_column: colorColumn || undefined,
            }),
        enabled: Boolean(datasetQuery.data?.loaded),
    });

    useEffect(() => {
        if (!geoQuery.data) return;
        if (!latTouched && !latColumn && geoQuery.data.lat_column) {
            setLatColumn(geoQuery.data.lat_column);
        }
        if (!lonTouched && !lonColumn && geoQuery.data.lon_column) {
            setLonColumn(geoQuery.data.lon_column);
        }
        if (!colorColumn && geoQuery.data.color_column) setColorColumn(geoQuery.data.color_column);
    }, [geoQuery.data, latColumn, lonColumn, colorColumn, latTouched, lonTouched]);

    const latOptions = useMemo(() => {
        const options = geoQuery.data?.candidates?.map((c) => c.lat_column) ?? [];
        return Array.from(new Set(options));
    }, [geoQuery.data?.candidates]);

    const lonOptions = useMemo(() => {
        const options = geoQuery.data?.candidates?.map((c) => c.lon_column) ?? [];
        return Array.from(new Set(options));
    }, [geoQuery.data?.candidates]);

    const colorOptions = useMemo(() => {
        const cols = datasetQuery.data?.column_names ?? [];
        return cols.filter((name) => name !== latColumn && name !== lonColumn);
    }, [datasetQuery.data?.column_names, latColumn, lonColumn]);

    if (geoQuery.isError) {
        return (
            <div className="flex items-center gap-2 rounded border border-red-500/40 bg-red-500/10 p-4 text-red-200">
                <AlertCircle className="h-5 w-5" />
                <span>Unable to build a geospatial view for this dataset.</span>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <ChartContainer
                title="Geospatial Distribution"
                description="Validates coordinates and projects the dataset on a natural-earth map"
                figure={geoQuery.data?.figure}
                isLoading={geoQuery.isLoading}
            />

            <section className="grid gap-4 rounded-lg border border-navy-500 bg-navy-700 p-4 md:grid-cols-3">
                <div className="space-y-1">
                    <label className="text-[10px] uppercase text-gray-400">Latitude Column</label>
                    <select
                        value={latColumn}
                        onChange={(e) => {
                            setLatTouched(true);
                            setLatColumn(e.target.value);
                        }}
                        className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
                    >
                        <option value="">Auto-detect</option>
                        {latOptions.map((candidate) => (
                            <option key={candidate} value={candidate}>
                                {candidate}
                            </option>
                        ))}
                    </select>
                </div>
                <div className="space-y-1">
                    <label className="text-[10px] uppercase text-gray-400">Longitude Column</label>
                    <select
                        value={lonColumn}
                        onChange={(e) => {
                            setLonTouched(true);
                            setLonColumn(e.target.value);
                        }}
                        className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
                    >
                        <option value="">Auto-detect</option>
                        {lonOptions.map((candidate) => (
                            <option key={candidate} value={candidate}>
                                {candidate}
                            </option>
                        ))}
                    </select>
                </div>
                <div className="space-y-1">
                    <label className="text-[10px] uppercase text-gray-400">Color Grouping</label>
                    <select
                        value={colorColumn}
                        onChange={(e) => setColorColumn(e.target.value)}
                        className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
                    >
                        <option value="">None</option>
                        {colorOptions.map((col) => (
                            <option key={col} value={col}>
                                {col}
                            </option>
                        ))}
                    </select>
                </div>
            </section>

            <section className="rounded-lg border border-navy-500 bg-navy-700 p-4 text-sm text-gray-200">
                <h4 className="text-sm font-semibold uppercase tracking-wider text-orange-500">
                    Coordinate Validation
                </h4>
                {geoQuery.isLoading && <Loader2 className="mt-2 h-4 w-4 animate-spin text-orange-500" />}
                {!geoQuery.isLoading && geoQuery.data && (
                    <div className="mt-3 grid gap-4 md:grid-cols-3">
                        <div>
                            <p className="text-[10px] uppercase text-gray-400">Valid Points</p>
                            <p className="text-xl font-semibold">{geoQuery.data.validation.valid_points.toLocaleString()}</p>
                        </div>
                        <div>
                            <p className="text-[10px] uppercase text-gray-400">Invalid Points</p>
                            <p className="text-xl font-semibold">{geoQuery.data.validation.invalid_points.toLocaleString()}</p>
                        </div>
                        <div>
                            <p className="text-[10px] uppercase text-gray-400">Validity %</p>
                            <p className="text-xl font-semibold">
                                {(geoQuery.data.validation.valid_ratio * 100).toFixed(1)}%
                            </p>
                        </div>
                    </div>
                )}
                {geoQuery.data?.warnings?.length ? (
                    <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-yellow-300">
                        {geoQuery.data.warnings.map((warning, idx) => (
                            <li key={idx}>{warning}</li>
                        ))}
                    </ul>
                ) : (
                    <p className="mt-3 text-xs text-gray-400">No coordinate issues detected.</p>
                )}
            </section>
        </div>
    );
}
