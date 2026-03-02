"use client";

import { useState, useCallback } from "react";

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
  const [isFocused, setIsFocused] = useState(false);

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
    if (!isFocused) {
      setIsFocused(true);
    }
    setDraft(text);

    const result = validate(text);
    if (result.valid) {
      setError("");
      onChange(result.num);
    } else {
      setError(result.message);
    }
  };

  const handleFocus = () => {
    setIsFocused(true);
    setDraft(String(value));
    setError("");
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
    setIsFocused(false);
  };

  return (
    <span className="number-input-wrapper">
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={isFocused ? draft : String(value)}
        onFocus={handleFocus}
        onChange={handleChange}
        onBlur={handleBlur}
        disabled={disabled}
        className={error ? "input-error" : ""}
      />
      {error && <span className="input-error-text">{error}</span>}
    </span>
  );
}
