export interface DataLoadResponse {
  dataset_name: string;
  rows: number;
  columns: number;
  column_names: string[];
}

export interface DatabaseConnectionResponse {
  connected: boolean;
  tables: string[];
  message: string;
}

export interface DatabaseConnectRequest {
  server: string;
  database: string;
}

export interface LoadTableRequest {
  table_name: string;
  limit?: number;
}

export interface RunQueryRequest {
  query: string;
}

export interface LoadSampleRequest {
  sample_type: string;
}

export interface TrendRequest {
  column: string;
}

export type FilterType = "numeric" | "categorical" | "datetime" | "boolean" | "text";

export interface FilterConditionPayload {
  column: string;
  type: FilterType;
  operator: string;
  value?: string | number;
  values?: Array<string | number>;
  range?: [string | number | null, string | number | null];
}

export interface ApplyFiltersRequest {
  filters: FilterConditionPayload[];
}

export interface CorrelationRequestPayload {
  col1: string;
  col2: string;
  method: "pearson" | "spearman" | "kendall";
}

export interface TimeSeriesRequestPayload {
  value_columns: string[];
  x_column?: string;
  chart_type?: "line" | "area" | "stacked";
  rolling_window?: number;
}

export interface GeospatialRequestPayload {
  lat_column?: string;
  lon_column?: string;
  color_column?: string;
}
