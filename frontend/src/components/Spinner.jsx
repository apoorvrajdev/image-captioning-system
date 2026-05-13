export default function Spinner({ size = "md", className = "" }) {
  const sizeMap = {
    sm: "w-4 h-4 border-2",
    md: "w-6 h-6 border-2",
    lg: "w-10 h-10 border-[3px]",
  };
  return (
    <div
      className={`${sizeMap[size]} ${className} rounded-full border-white/15 border-t-violet-400 animate-spin`}
      role="status"
      aria-label="Loading"
    />
  );
}
