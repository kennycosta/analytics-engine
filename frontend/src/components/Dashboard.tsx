import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import Header from "./Header";
import Sidebar from "./Sidebar";
import MainContent from "./MainContent";
import FilterPanel from "./filters/FilterPanel";
import { BarChart3 } from "lucide-react";
import type { TabKey } from "./tabs/tabConfig";

export default function Dashboard() {
  const queryClient = useQueryClient();
  const [hasData, setHasData] = useState(false);
  const [activeTab, setActiveTab] = useState<TabKey>("overview");

  function handleDataLoaded() {
    setHasData(true);
    // Invalidate all queries to refetch with new data
    queryClient.invalidateQueries();
  }

  return (
    <div className="flex h-screen bg-navy-600">
      <Sidebar
        onDataLoaded={handleDataLoaded}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />
      <div className="flex flex-1 flex-col border-r border-navy-500">
        <Header />
        <div className="flex flex-1 overflow-hidden">
          {hasData ? (
            <MainContent activeTab={activeTab} />
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center gap-4 text-gray-400">
              <BarChart3 className="h-16 w-16 text-navy-400" />
              <p className="text-lg">Load a dataset to get started</p>
              <p className="text-sm text-gray-500">
                Upload a file, connect to SQL, or choose a sample dataset
              </p>
            </div>
          )}
          <FilterPanel isDisabled={!hasData} />
        </div>
      </div>
    </div>
  );
}
