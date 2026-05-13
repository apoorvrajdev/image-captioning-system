import { useEffect, useRef, useState } from "react";
import { checkHealth } from "../services/api";

const POLL_INTERVAL_MS = 10000;

export default function StatusBadge() {
  const [health, setHealth] = useState(null);
  const [phase, setPhase] = useState("checking"); // 'checking' | 'online' | 'offline'
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    let cancelled = false;

    const poll = async () => {
      const result = await checkHealth();
      if (cancelled || !mountedRef.current) return;
      if (result?.model_loaded) {
        setHealth(result);
        setPhase("online");
      } else {
        setHealth(result);
        setPhase("offline");
      }
    };

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);

    const onFocus = () => poll();
    window.addEventListener("focus", onFocus);

    return () => {
      cancelled = true;
      mountedRef.current = false;
      clearInterval(interval);
      window.removeEventListener("focus", onFocus);
    };
  }, []);

  if (phase === "checking") {
    return (
      <div className="inline-flex items-center gap-2 px-3 py-1 bg-gray-700 text-gray-300 rounded-full text-sm">
        <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse" />
        Checking...
      </div>
    );
  }

  const isOnline = phase === "online";
  const statusColor = isOnline ? "bg-green-500" : "bg-red-500";
  const statusText = isOnline ? "Backend online" : "Backend offline";

  return (
    <div className="inline-flex items-center gap-2 px-3 py-1 bg-gray-900 border border-gray-700 rounded-full text-sm">
      <div className={`w-2 h-2 ${statusColor} rounded-full`} />
      <span className="text-gray-300">{statusText}</span>
      {health?.api_version && (
        <span className="text-gray-500 text-xs ml-1">
          v{health.api_version}
        </span>
      )}
    </div>
  );
}
