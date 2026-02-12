import {
    LayoutDashboard,
    GitBranch,
    TrendingUp,
    Brain,
    Globe2,
} from "lucide-react";

export const DASHBOARD_TABS = [
    { key: "overview", label: "Overview", icon: LayoutDashboard },
    { key: "relationships", label: "Relationships", icon: GitBranch },
    { key: "trends", label: "Trends", icon: TrendingUp },
    { key: "insights", label: "Insights", icon: Brain },
    { key: "geospatial", label: "Geospatial", icon: Globe2 },
] as const;

export type TabKey = (typeof DASHBOARD_TABS)[number]["key"];
