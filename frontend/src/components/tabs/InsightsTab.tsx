import { useQuery } from "@tanstack/react-query";
import { getInsights } from "@/lib/api";
import { Loader2, AlertCircle } from "lucide-react";

export default function InsightsTab() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["insights"],
    queryFn: getInsights,
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
        <span>Failed to generate insights</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-orange-500">
        AI-Generated Narrative
      </h3>

      <div className="prose prose-invert max-w-none rounded-lg border border-navy-500 bg-navy-700 p-6 text-gray-300 leading-relaxed whitespace-pre-line">
        {data.narrative}
      </div>

      {data.correlations.length > 0 && (
        <section>
          <h4 className="mb-2 text-sm font-semibold text-orange-500">
            Detected Correlations
          </h4>
          <div className="overflow-x-auto rounded border border-navy-500">
            <table className="min-w-full text-sm">
              <thead className="bg-navy-700 text-xs uppercase text-gray-400">
                <tr>
                  <th className="px-3 py-2 text-left">Var 1</th>
                  <th className="px-3 py-2 text-left">Var 2</th>
                  <th className="px-3 py-2 text-right">r</th>
                  <th className="px-3 py-2 text-right">p-value</th>
                  <th className="px-3 py-2 text-left">Strength</th>
                </tr>
              </thead>
              <tbody>
                {data.correlations.map((c, i) => (
                  <tr key={i} className="border-t border-navy-500 hover:bg-navy-700/50">
                    <td className="px-3 py-1.5 text-gray-300">{c.variable1}</td>
                    <td className="px-3 py-1.5 text-gray-300">{c.variable2}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-gray-300">
                      {c.correlation.toFixed(3)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-gray-300">
                      {c.p_value.toFixed(4)}
                    </td>
                    <td className="px-3 py-1.5">
                      <span
                        className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${
                          c.strength === "strong"
                            ? "bg-orange-500/20 text-orange-400"
                            : c.strength === "moderate"
                              ? "bg-yellow-500/20 text-yellow-400"
                              : "bg-gray-500/20 text-gray-400"
                        }`}
                      >
                        {c.strength}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
