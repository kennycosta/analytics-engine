import { useState } from "react";
import { FlaskConical } from "lucide-react";
import { loadSampleData } from "@/lib/api";

interface Props {
  onLoaded: () => void;
}

const SAMPLES = [
  { type: "sales", label: "Sales" },
  { type: "customer", label: "Customer" },
  { type: "random", label: "Random" },
] as const;

export default function SampleData({ onLoaded }: Props) {
  const [loading, setLoading] = useState<string | null>(null);
  const [loaded, setLoaded] = useState<string | null>(null);

  async function handleLoad(sampleType: string) {
    setLoading(sampleType);
    try {
      const res = await loadSampleData({ sample_type: sampleType });
      setLoaded(`${res.dataset_name} (${res.rows} rows)`);
      onLoaded();
    } catch {
      setLoaded(null);
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-gray-300">
        Sample Dataset
      </label>
      <div className="flex flex-col gap-2">
        {SAMPLES.map((s) => (
          <button
            key={s.type}
            onClick={() => handleLoad(s.type)}
            disabled={loading !== null}
            className="flex items-center justify-center gap-2 rounded bg-navy-500 px-4 py-2 text-sm font-medium text-white hover:bg-navy-400 disabled:opacity-50"
          >
            <FlaskConical className="h-4 w-4 text-orange-500" />
            {loading === s.type ? "Loading..." : s.label}
          </button>
        ))}
      </div>
      {loaded && <p className="text-xs text-green-400">Loaded: {loaded}</p>}
    </div>
  );
}
