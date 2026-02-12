import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
    getCurrentDataset,
    getFilterOptions,
    applyDatasetFilters,
    resetDatasetFilters,
} from "@/lib/api";
import type { FilterConditionPayload } from "@/types/api";
import type { FilterOption } from "@/types/dataset";
import {
    Filter as FilterIcon,
    SlidersHorizontal,
    Search,
    X,
    RefreshCcw,
} from "lucide-react";

interface Props {
    isDisabled: boolean;
}

const NUMERIC_OPERATORS = ["between", ">=", "<=", "equals"];
const CATEGORICAL_OPERATORS = ["equals", "in", "contains"];
const DATETIME_OPERATORS = ["between", "before", "after"];

export default function FilterPanel({ isDisabled }: Props) {
    const queryClient = useQueryClient();
    const { data: datasetInfo } = useQuery({
        queryKey: ["currentDataset"],
        queryFn: getCurrentDataset,
        enabled: !isDisabled,
        staleTime: 15_000,
    });

    const { data: filterOptions } = useQuery({
        queryKey: ["filterOptions", datasetInfo?.filtered_rows],
        queryFn: getFilterOptions,
        enabled: !isDisabled && Boolean(datasetInfo?.loaded),
        staleTime: 15_000,
    });

    const [selectedColumn, setSelectedColumn] = useState<string>("");
    const [operator, setOperator] = useState<string>("between");
    const [numericRange, setNumericRange] = useState<[number | null, number | null]>([
        null,
        null,
    ]);
    const [dateRange, setDateRange] = useState<[string | null, string | null]>([null, null]);
    const [scalarValue, setScalarValue] = useState<string>("");
    const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
    const [categorySearch, setCategorySearch] = useState("");

    const activeOption = useMemo<FilterOption | undefined>(() => {
        return filterOptions?.columns.find((c) => c.name === selectedColumn);
    }, [filterOptions, selectedColumn]);

    useEffect(() => {
        if (filterOptions?.columns.length && !selectedColumn) {
            setSelectedColumn(filterOptions.columns[0].name);
        }
    }, [filterOptions, selectedColumn]);

    useEffect(() => {
        if (!activeOption) {
            return;
        }
        const type = normalizeType(activeOption.detected_type);
        if (type === "numeric" && activeOption.numeric_range) {
            setNumericRange([activeOption.numeric_range.min, activeOption.numeric_range.max]);
            setOperator("between");
        } else if (type === "datetime") {
            setDateRange([null, null]);
            setOperator("between");
        } else {
            setOperator(type === "categorical" ? "equals" : "contains");
        }
        setScalarValue("");
        setSelectedCategories([]);
        setCategorySearch("");
    }, [activeOption]);

    const applyMutation = useMutation({
        mutationFn: (filters: FilterConditionPayload[]) =>
            applyDatasetFilters({ filters }),
        onSuccess: () => {
            queryClient.invalidateQueries();
        },
    });

    const resetMutation = useMutation({
        mutationFn: resetDatasetFilters,
        onSuccess: () => {
            queryClient.invalidateQueries();
        },
    });

    const currentFilters = datasetInfo?.active_filters ?? [];

    function handleRemoveFilter(index: number) {
        const next = currentFilters.filter((_, i) => i !== index);
        applyMutation.mutate(next);
    }

    function handleApplyNew() {
        if (!activeOption || !selectedColumn) {
            return;
        }
        const type = normalizeType(activeOption.detected_type);
        const next: FilterConditionPayload = {
            column: selectedColumn,
            type,
            operator,
        };

        if (type === "numeric") {
            if (operator === "between") {
                next.range = numericRange;
            } else {
                next.value = scalarValue || (numericRange[0] ?? undefined);
            }
        } else if (type === "datetime") {
            if (operator === "between") {
                next.range = dateRange;
            } else {
                next.value = scalarValue;
            }
        } else {
            if (operator === "in") {
                next.values = selectedCategories;
            } else {
                next.value = scalarValue || selectedCategories[0];
            }
        }

        applyMutation.mutate([...currentFilters, next]);
    }

    const disabledReason = !datasetInfo?.loaded
        ? "Load data to enable filters"
        : isDisabled
            ? ""
            : null;

    const filteredCategories = (activeOption?.categorical_values ?? []).filter((val) =>
        val.toLowerCase().includes(categorySearch.toLowerCase())
    );

    const operators = useMemo(() => {
        const type = normalizeType(activeOption?.detected_type);
        if (type === "numeric") return NUMERIC_OPERATORS;
        if (type === "datetime") return DATETIME_OPERATORS;
        return CATEGORICAL_OPERATORS;
    }, [activeOption?.detected_type]);

    const numericMin = activeOption?.numeric_range?.min ?? 0;
    const numericMax = activeOption?.numeric_range?.max ?? 0;

    return (
        <aside className="w-80 border-l border-navy-500 bg-navy-700 p-5 text-sm">
            <div className="mb-4 flex items-center justify-between">
                <div>
                    <p className="text-xs uppercase tracking-wider text-gray-400">Rows</p>
                    <p className="text-xl font-semibold">
                        {datasetInfo?.filtered_rows?.toLocaleString() ?? "—"}
                        {datasetInfo?.rows && datasetInfo.filtered_rows !== datasetInfo.rows && (
                            <span className="ml-1 text-xs text-gray-400">
                                / {datasetInfo.rows.toLocaleString()}
                            </span>
                        )}
                    </p>
                </div>
                <button
                    className="rounded border border-navy-500 px-2 py-1 text-xs text-gray-300 transition-colors hover:border-orange-500 hover:text-white"
                    onClick={() => resetMutation.mutate()}
                    disabled={resetMutation.isPending || !currentFilters.length}
                >
                    <RefreshCcw className="mr-1 inline h-3.5 w-3.5" />
                    Reset
                </button>
            </div>

            {disabledReason && (
                <div className="rounded border border-dashed border-navy-500 bg-navy-600 p-3 text-center text-xs text-gray-400">
                    {disabledReason}
                </div>
            )}

            {!disabledReason && (
                <>
                    <section className="space-y-2">
                        <div className="flex items-center justify-between">
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-orange-500">
                                Active Filters
                            </h3>
                            <FilterIcon className="h-4 w-4 text-orange-500" />
                        </div>
                        {currentFilters.length === 0 ? (
                            <p className="text-xs text-gray-400">No filters applied</p>
                        ) : (
                            <div className="flex flex-wrap gap-2">
                                {currentFilters.map((f, idx) => (
                                    <span
                                        key={`${f.column}-${idx}`}
                                        className="inline-flex items-center gap-1 rounded-full bg-navy-600 px-3 py-1 text-xs"
                                    >
                                        {f.column}: {f.operator}
                                        <button
                                            className="text-gray-400 hover:text-orange-400"
                                            onClick={() => handleRemoveFilter(idx)}
                                        >
                                            <X className="h-3 w-3" />
                                        </button>
                                    </span>
                                ))}
                            </div>
                        )}
                    </section>

                    <section className="mt-5 space-y-3">
                        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-orange-500">
                            <SlidersHorizontal className="h-4 w-4" />
                            Configure Filter
                        </div>

                        <div className="space-y-2">
                            <label className="text-[10px] uppercase text-gray-400">Column</label>
                            <select
                                value={selectedColumn}
                                onChange={(e) => setSelectedColumn(e.target.value)}
                                className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
                            >
                                {filterOptions?.columns.map((col) => (
                                    <option key={col.name} value={col.name}>
                                        {col.name}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="space-y-2">
                            <label className="text-[10px] uppercase text-gray-400">Operator</label>
                            <select
                                value={operator}
                                onChange={(e) => setOperator(e.target.value)}
                                className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
                            >
                                {operators.map((op) => (
                                    <option key={op} value={op}>
                                        {op}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {normalizeType(activeOption?.detected_type) === "numeric" && (
                            <NumericFilterControls
                                min={numericMin}
                                max={numericMax}
                                range={numericRange}
                                onRangeChange={setNumericRange}
                                operator={operator}
                                scalarValue={scalarValue}
                                onScalarChange={setScalarValue}
                            />
                        )}

                        {normalizeType(activeOption?.detected_type) === "datetime" && (
                            <DateFilterControls
                                range={dateRange}
                                onRangeChange={setDateRange}
                                operator={operator}
                                scalarValue={scalarValue}
                                onScalarChange={setScalarValue}
                            />
                        )}

                        {normalizeType(activeOption?.detected_type) === "categorical" && (
                            <CategoricalFilterControls
                                values={filteredCategories}
                                selected={selectedCategories}
                                onToggle={(val, isRadio) =>
                                    toggleCategory(val, selectedCategories, setSelectedCategories, isRadio)
                                }
                                search={categorySearch}
                                onSearch={setCategorySearch}
                                operator={operator}
                                scalarValue={scalarValue}
                                onScalarChange={setScalarValue}
                            />
                        )}

                        <button
                            className="w-full rounded bg-orange-500 py-2 text-sm font-semibold text-black transition-colors disabled:cursor-not-allowed disabled:bg-orange-500/40"
                            onClick={handleApplyNew}
                            disabled={applyMutation.isPending || !selectedColumn}
                        >
                            Apply Filter
                        </button>
                    </section>
                </>
            )}
        </aside>
    );
}

function normalizeType(raw?: string): "numeric" | "categorical" | "datetime" {
    if (!raw) return "categorical";
    if (raw.includes("numeric") || raw.includes("number")) return "numeric";
    if (raw.includes("date")) return "datetime";
    return "categorical";
}

function toggleCategory(
    value: string,
    selected: string[],
    setter: (values: string[]) => void,
    asRadio = false
) {
    if (asRadio) {
        setter([value]);
        return;
    }
    if (selected.includes(value)) {
        setter(selected.filter((v) => v !== value));
    } else {
        setter([...selected, value]);
    }
}

interface NumericProps {
    min: number;
    max: number;
    range: [number | null, number | null];
    onRangeChange: (range: [number | null, number | null]) => void;
    operator: string;
    scalarValue: string;
    onScalarChange: (val: string) => void;
}

function NumericFilterControls({
    min,
    max,
    range,
    onRangeChange,
    operator,
    scalarValue,
    onScalarChange,
}: NumericProps) {
    const presets: Array<[string, [number, number]]> = [
        ["Full", [min, max]],
        ["Mid 50%", [min + (max - min) * 0.25, min + (max - min) * 0.75]],
        ["Upper", [min + (max - min) * 0.5, max]],
    ];

    return (
        <div className="space-y-3">
            {operator === "between" ? (
                <>
                    <div className="flex items-center justify-between text-[10px] uppercase text-gray-400">
                        <span>Range</span>
                        <div className="flex gap-2">
                            {presets.map(([label, preset]) => (
                                <button
                                    key={label}
                                    className="rounded border border-navy-500 px-2 py-0.5 text-xs text-gray-300 hover:border-orange-500 hover:text-white"
                                    onClick={() => onRangeChange([preset[0], preset[1]])}
                                    type="button"
                                >
                                    {label}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="space-y-2">
                        <input
                            type="range"
                            min={min}
                            max={max}
                            value={range[0] ?? min}
                            onChange={(e) => onRangeChange([Number(e.target.value), range[1]])}
                            className="w-full"
                        />
                        <input
                            type="range"
                            min={min}
                            max={max}
                            value={range[1] ?? max}
                            onChange={(e) => onRangeChange([range[0], Number(e.target.value)])}
                            className="w-full"
                        />
                        <div className="flex items-center gap-2 text-xs text-gray-400">
                            <span>{range[0] != null ? range[0].toFixed(2) : "—"}</span>
                            <span className="text-gray-500">to</span>
                            <span>{range[1] != null ? range[1].toFixed(2) : "—"}</span>
                        </div>
                    </div>
                </>
            ) : (
                <div className="space-y-2">
                    <label className="text-[10px] uppercase text-gray-400">Value</label>
                    <input
                        type="number"
                        value={scalarValue}
                        onChange={(e) => onScalarChange(e.target.value)}
                        className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
                    />
                </div>
            )}
        </div>
    );
}

interface DateProps {
    range: [string | null, string | null];
    onRangeChange: (range: [string | null, string | null]) => void;
    operator: string;
    scalarValue: string;
    onScalarChange: (val: string) => void;
}

function DateFilterControls({
    range,
    onRangeChange,
    operator,
    scalarValue,
    onScalarChange,
}: DateProps) {
    return (
        <div className="space-y-2">
            {operator === "between" ? (
                <>
                    <label className="text-[10px] uppercase text-gray-400">Date Range</label>
                    <input
                        type="date"
                        value={range[0] ?? ""}
                        onChange={(e) => onRangeChange([e.target.value || null, range[1]])}
                        className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
                    />
                    <input
                        type="date"
                        value={range[1] ?? ""}
                        onChange={(e) => onRangeChange([range[0], e.target.value || null])}
                        className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
                    />
                </>
            ) : (
                <>
                    <label className="text-[10px] uppercase text-gray-400">Date</label>
                    <input
                        type="date"
                        value={scalarValue}
                        onChange={(e) => onScalarChange(e.target.value)}
                        className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
                    />
                </>
            )}
        </div>
    );
}

interface CategoricalProps {
    values: string[];
    selected: string[];
    onToggle: (value: string, asRadio: boolean) => void;
    search: string;
    onSearch: (value: string) => void;
    operator: string;
    scalarValue: string;
    onScalarChange: (val: string) => void;
}

function CategoricalFilterControls({
    values,
    selected,
    onToggle,
    search,
    onSearch,
    operator,
    scalarValue,
    onScalarChange,
}: CategoricalProps) {
    return (
        <div className="space-y-2">
            {operator === "contains" ? (
                <div className="space-y-1">
                    <label className="text-[10px] uppercase text-gray-400">Search Text</label>
                    <input
                        type="text"
                        value={scalarValue}
                        onChange={(e) => onScalarChange(e.target.value)}
                        className="w-full rounded border border-navy-500 bg-navy-600 px-2 py-2 text-sm focus:border-orange-500 focus:outline-none"
                    />
                </div>
            ) : (
                <>
                    <div className="flex items-center gap-2 rounded border border-navy-500 bg-navy-600 px-2 py-1.5">
                        <Search className="h-4 w-4 text-gray-400" />
                        <input
                            type="text"
                            value={search}
                            onChange={(e) => onSearch(e.target.value)}
                            placeholder="Search categories"
                            className="flex-1 bg-transparent text-sm focus:outline-none"
                        />
                    </div>
                    <div className="max-h-32 space-y-1 overflow-y-auto">
                        {values.map((val) => (
                            <label key={val} className="flex items-center gap-2 text-sm text-gray-200">
                                <input
                                    type={operator === "equals" ? "radio" : "checkbox"}
                                    name={operator === "equals" ? "categorical-equals" : undefined}
                                    checked={selected.includes(val)}
                                    onChange={() => onToggle(val, operator === "equals")}
                                />
                                {val}
                            </label>
                        ))}
                        {values.length === 0 && (
                            <p className="text-xs text-gray-500">No categories match your search</p>
                        )}
                    </div>
                </>
            )}
        </div>
    );
}
