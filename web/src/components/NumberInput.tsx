"use client";

import { useState, useEffect, useCallback } from "react";

type NumberInputProps = {
  value: number;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  onChange: (value: number) => void;
};

export function NumberInput({
  value,
  min,
  max,
  step,
  disabled,
  onChange,
}: NumberInputProps) {
  const [draft, setDraft] = useState(String(value));
  const [error, setError] = useState("");

  // Sync draft when value changes externally (e.g. from parent state)
  useEffect(() => {
    setDraft(String(value));
    setError("");
  }, [value]);

  const validate = useCallback(
    (text: string): { valid: boolean; num: number; message: string } => {
      if (text.trim() === "") {
        return { valid: false, num: 0, message: `Required (${min}–${max})` };
      }
      const num = Number(text);
      if (!Number.isFinite(num) || text.trim() !== String(num)) {
        return { valid: false, num: 0, message: "Must be a number" };
      }
      if (num < min || num > max) {
        return {
          valid: false,
          num,
          message: `Must be ${min}–${max}`,
        };
      }
      return { valid: true, num, message: "" };
    },
    [min, max],
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const text = e.target.value;
    setDraft(text);

    const result = validate(text);
    if (result.valid) {
      setError("");
      onChange(result.num);
    } else {
      setError(result.message);
    }
  };

  const handleBlur = () => {
    const result = validate(draft);
    if (result.valid) {
      setError("");
      onChange(result.num);
    } else {
      // Revert to last valid value
      setDraft(String(value));
      setError("");
    }
  };

  return (
    <span className="number-input-wrapper">
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={draft}
        onChange={handleChange}
        onBlur={handleBlur}
        disabled={disabled}
        className={error ? "input-error" : ""}
      />
      {error && <span className="input-error-text">{error}</span>}
    </span>
  );
}
