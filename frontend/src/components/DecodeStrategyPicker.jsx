// Greedy / Beam search choice for the next request (TASK-034). Native radio
// inputs, so arrow keys move between the two and the choice is announced.

const OPTIONS = [
  {
    id: "greedy",
    label: "Greedy",
    hint: "Fastest. The paper's decoder.",
  },
  {
    id: "beam",
    label: "Beam search",
    hint: "Width 4. Slower, higher CIDEr on our evaluation.",
  },
];

export default function DecodeStrategyPicker({ value, onChange, disabled }) {
  return (
    <fieldset disabled={disabled} className="mb-4">
      <legend className="text-xs uppercase tracking-wider text-white/50 mb-2">
        Decoding
      </legend>
      <div
        className="grid grid-cols-2 gap-1 rounded-xl border border-white/10 bg-white/[0.02] p-1"
        role="presentation"
      >
        {OPTIONS.map((option) => {
          const checked = option.id === value;
          return (
            <label
              key={option.id}
              className={[
                "cursor-pointer rounded-lg px-3 py-2 text-left transition-colors",
                "has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-fuchsia-300",
                checked
                  ? "bg-white/10 text-white"
                  : "text-white/50 hover:text-white/80",
                disabled ? "cursor-not-allowed opacity-60" : "",
              ].join(" ")}
            >
              <input
                type="radio"
                name="decode-strategy"
                value={option.id}
                checked={checked}
                onChange={() => onChange(option.id)}
                className="sr-only"
              />
              <span className="block text-sm font-medium">{option.label}</span>
              <span className="block text-[11px] text-white/50 mt-0.5">
                {option.hint}
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
