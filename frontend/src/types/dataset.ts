import type { FilterConditionPayload } from "./api";

export interface DatasetInfo {
  loaded: boolean;
  dataset_name?: string;
  rows?: number;
  filtered_rows?: number;
  columns?: number;
  column_names?: string[];
  numeric_columns?: string[];
  categorical_columns?: string[];
  filters_active?: boolean;
  active_filters?: FilterConditionPayload[];
}

export interface DatasetOverview {
  name: string;
  row_count: number;
  column_count: number;
  memory_usage_mb: number;
  completeness_score: number;
  column_types: Record<string, number>;
  quality_issues: string[];
  insights: string[];
  preview: Record<string, unknown>[];
  describe: DescribeRow[];
  summary: string;
}

export interface DescribeRow {
  column: string;
  stats: Record<string, string | number | null>;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  detected_type: string;
  count: number;
  null_count: number;
  null_percentage: number;
  unique_count: number;
  unique_percentage: number;
  numeric_stats?: Record<string, number>;
  categorical_stats?: Record<string, unknown>;
  datetime_stats?: Record<string, string>;
  outlier_count?: number;
  quality_issues?: string[];
  insights: string[];
}

export interface CorrelationMatrix {
  columns: string[];
  matrix: number[][];
}

export interface CorrelationResult {
  variable1: string;
  variable2: string;
  correlation: number;
  p_value: number;
  method: string;
  significant: boolean;
  strength: string;
}

export interface TrendResult {
  column: string;
  trend: string;
  slope?: number;
  r_squared?: number;
  p_value?: number;
  significant?: boolean;
}

export interface InsightsResult {
  narrative: string;
  correlations: {
    variable1: string;
    variable2: string;
    correlation: number;
    p_value: number;
    significant: boolean;
    strength: string;
  }[];
}

export interface PlotlyFigure {
  figure: {
    data: unknown[];
    layout: Record<string, unknown>;
  };
}

export interface FilterOption {
  name: string;
  detected_type: string;
  searchable: boolean;
  categorical_values?: string[];
  numeric_range?: { min: number; max: number };
  datetime_range?: { start: string | null; end: string | null };
}

export interface FilterOptionsPayload {
  columns: FilterOption[];
}

export interface FilterApplicationResult {
  row_count: number;
  column_count: number;
  preview: Record<string, unknown>[];
  active_filters: FilterConditionPayload[];
}

export interface GeospatialResponsePayload {
  figure: PlotlyFigure["figure"];
  lat_column: string;
  lon_column: string;
  color_column?: string | null;
  validation: {
    valid_points: number;
    invalid_points: number;
    valid_ratio: number;
  };
  warnings: string[];
  candidates: {
    lat_column: string;
    lon_column: string;
    lat_valid_ratio: number;
    lon_valid_ratio: number;
  }[];
}
