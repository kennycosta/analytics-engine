import { useState } from "react";
import { Upload, Database, FlaskConical } from "lucide-react";
import FileUpload from "./sidebar/FileUpload";
import SqlConnection from "./sidebar/SqlConnection";
import SampleData from "./sidebar/SampleData";
import { DASHBOARD_TABS, type TabKey } from "./tabs/tabConfig";

type Source = "file" | "sql" | "sample";

const SOURCES: { key: Source; label: string; icon: React.ReactNode }[] = [
  { key: "file", label: "Upload", icon: <Upload className="h-4 w-4" /> },
  { key: "sql", label: "SQL", icon: <Database className="h-4 w-4" /> },
  { key: "sample", label: "Sample", icon: <FlaskConical className="h-4 w-4" /> },
];

interface Props {
  onDataLoaded: () => void;
  activeTab: TabKey;
  onTabChange: (tab: TabKey) => void;
}

export default function Sidebar({ onDataLoaded, activeTab, onTabChange }: Props) {
  const [source, setSource] = useState<Source>("sample");

  return (
    <aside className="flex w-80 flex-col gap-6 border-r border-navy-500 bg-navy-700 p-5 overflow-y-auto">
      <section className="space-y-2">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-orange-500">
          Navigate
        </h2>
        <nav className="flex flex-col gap-1">
          {DASHBOARD_TABS.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => onTabChange(key)}
              className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${activeTab === key
                  ? "bg-orange-500 text-black"
                  : "text-gray-300 hover:text-white"
                }`}
            >
              <Icon className={`h-4 w-4 ${activeTab === key ? "text-black" : "text-orange-500"}`} />
              {label}
            </button>
          ))}
        </nav>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-orange-500">
          Data Source
        </h2>

        <div className="flex rounded bg-navy-600">
          {SOURCES.map((s) => (
            <button
              key={s.key}
              onClick={() => setSource(s.key)}
              className={`flex flex-1 items-center justify-center gap-1.5 rounded px-2 py-2 text-xs font-medium transition-colors ${source === s.key
                  ? "bg-orange-500 text-black"
                  : "text-gray-300 hover:text-white"
                }`}
            >
              {s.icon}
              {s.label}
            </button>
          ))}
        </div>

        {source === "file" && <FileUpload onLoaded={onDataLoaded} />}
        {source === "sql" && <SqlConnection onLoaded={onDataLoaded} />}
        {source === "sample" && <SampleData onLoaded={onDataLoaded} />}
      </section>
    </aside>
  );
}
