import { useState } from "react";
import { Database } from "lucide-react";
import { connectDatabase, loadTable, runQuery } from "@/lib/api";

interface Props {
  onLoaded: () => void;
}

export default function SqlConnection({ onLoaded }: Props) {
  const [server, setServer] = useState("ekofisk");
  const [database, setDatabase] = useState("");
  const [connected, setConnected] = useState(false);
  const [tables, setTables] = useState<string[]>([]);
  const [selectedTable, setSelectedTable] = useState("");
  const [rowLimit, setRowLimit] = useState(5000);
  const [mode, setMode] = useState<"table" | "query">("table");
  const [sql, setSql] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  async function handleConnect() {
    setLoading(true);
    setError(null);
    try {
      const res = await connectDatabase({ server, database });
      setConnected(res.connected);
      setTables(res.tables);
      if (res.tables.length > 0) setSelectedTable(res.tables[0]);
      setStatus(res.message);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Connection failed";
      setError(msg);
      setConnected(false);
    } finally {
      setLoading(false);
    }
  }

  async function handleLoadTable() {
    if (!selectedTable) return;
    setLoading(true);
    setError(null);
    try {
      const res = await loadTable({ table_name: selectedTable, limit: rowLimit });
      setStatus(`Loaded ${res.rows} rows`);
      onLoaded();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Load failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleRunQuery() {
    if (!sql.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runQuery({ query: sql });
      setStatus(`Query returned ${res.rows} rows`);
      onLoaded();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Query failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-gray-300">Server</label>
      <input
        value={server}
        onChange={(e) => setServer(e.target.value)}
        className="w-full rounded border border-navy-500 bg-navy-600 px-3 py-1.5 text-sm text-white focus:border-orange-500 focus:outline-none"
      />

      <label className="text-sm font-medium text-gray-300">Database</label>
      <input
        value={database}
        onChange={(e) => setDatabase(e.target.value)}
        className="w-full rounded border border-navy-500 bg-navy-600 px-3 py-1.5 text-sm text-white focus:border-orange-500 focus:outline-none"
      />

      <button
        onClick={handleConnect}
        disabled={loading || !server || !database}
        className="flex w-full items-center justify-center gap-2 rounded bg-orange-500 px-4 py-2 text-sm font-semibold text-black hover:bg-orange-400 disabled:opacity-50"
      >
        <Database className="h-4 w-4" />
        {loading && !connected ? "Connecting..." : "Connect"}
      </button>

      {connected && (
        <>
          <div className="border-t border-navy-500 pt-3">
            <div className="flex gap-2 text-sm mb-2">
              <button
                onClick={() => setMode("table")}
                className={`rounded px-3 py-1 ${mode === "table" ? "bg-orange-500 text-black" : "bg-navy-600 text-gray-300"}`}
              >
                Table
              </button>
              <button
                onClick={() => setMode("query")}
                className={`rounded px-3 py-1 ${mode === "query" ? "bg-orange-500 text-black" : "bg-navy-600 text-gray-300"}`}
              >
                SQL Query
              </button>
            </div>

            {mode === "table" ? (
              <>
                <select
                  value={selectedTable}
                  onChange={(e) => setSelectedTable(e.target.value)}
                  className="w-full rounded border border-navy-500 bg-navy-600 px-3 py-1.5 text-sm text-white focus:outline-none"
                >
                  {tables.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
                <label className="text-xs text-gray-400 mt-1 block">
                  Row limit
                </label>
                <input
                  type="number"
                  value={rowLimit}
                  onChange={(e) => setRowLimit(Number(e.target.value))}
                  min={100}
                  max={100000}
                  step={500}
                  className="w-full rounded border border-navy-500 bg-navy-600 px-3 py-1.5 text-sm text-white focus:outline-none"
                />
                <button
                  onClick={handleLoadTable}
                  disabled={loading}
                  className="mt-2 w-full rounded bg-orange-500 px-4 py-2 text-sm font-semibold text-black hover:bg-orange-400 disabled:opacity-50"
                >
                  {loading ? "Loading..." : "Load Table"}
                </button>
              </>
            ) : (
              <>
                <textarea
                  value={sql}
                  onChange={(e) => setSql(e.target.value)}
                  rows={5}
                  placeholder="SELECT * FROM ..."
                  className="w-full rounded border border-navy-500 bg-navy-600 px-3 py-2 text-sm text-white font-mono focus:outline-none"
                />
                <button
                  onClick={handleRunQuery}
                  disabled={loading || !sql.trim()}
                  className="mt-2 w-full rounded bg-orange-500 px-4 py-2 text-sm font-semibold text-black hover:bg-orange-400 disabled:opacity-50"
                >
                  {loading ? "Running..." : "Run Query"}
                </button>
              </>
            )}
          </div>
        </>
      )}

      {status && <p className="text-xs text-green-400">{status}</p>}
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}
