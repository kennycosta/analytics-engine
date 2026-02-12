import OverviewTab from "./tabs/OverviewTab";
import RelationshipsTab from "./tabs/RelationshipsTab";
import TrendsTab from "./tabs/TrendsTab";
import InsightsTab from "./tabs/InsightsTab";
import GeospatialTab from "./tabs/GeospatialTab";
import type { TabKey } from "./tabs/tabConfig";

interface Props {
  activeTab: TabKey;
}

export default function MainContent({ activeTab }: Props) {
  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-6">
        {activeTab === "overview" && <OverviewTab />}
        {activeTab === "relationships" && <RelationshipsTab />}
        {activeTab === "trends" && <TrendsTab />}
        {activeTab === "insights" && <InsightsTab />}
        {activeTab === "geospatial" && <GeospatialTab />}
      </div>
    </div>
  );
}
