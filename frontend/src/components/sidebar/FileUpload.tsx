import { useState, useRef } from "react";
import { Upload } from "lucide-react";
import { uploadFile } from "@/lib/api";

interface Props {
  onLoaded: () => void;
}

export default function FileUpload({ onLoaded }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  async function handleUpload() {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      await uploadFile(file);
      setFileName(file.name);
      onLoaded();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Upload failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-gray-300">
        Upload CSV or Excel
      </label>
      <input
        ref={fileRef}
        type="file"
        accept=".csv,.xlsx,.xls"
        className="block w-full text-sm text-gray-300 file:mr-3 file:rounded file:border-0 file:bg-navy-500 file:px-3 file:py-1.5 file:text-sm file:text-white hover:file:bg-navy-400"
      />
      <button
        onClick={handleUpload}
        disabled={loading}
        className="flex w-full items-center justify-center gap-2 rounded bg-orange-500 px-4 py-2 text-sm font-semibold text-black hover:bg-orange-400 disabled:opacity-50"
      >
        <Upload className="h-4 w-4" />
        {loading ? "Uploading..." : "Upload"}
      </button>
      {fileName && (
        <p className="text-xs text-green-400">Loaded: {fileName}</p>
      )}
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}
