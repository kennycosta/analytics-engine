import { BarChart3 } from "lucide-react";

export default function Header() {
  return (
    <header className="flex items-center gap-3 bg-navy-700 px-6 py-4 border-b border-navy-500">
      <BarChart3 className="h-7 w-7 text-orange-500" />
      <h1 className="text-xl font-bold tracking-tight">
        AI Analytics Engine
      </h1>
    </header>
  );
}
