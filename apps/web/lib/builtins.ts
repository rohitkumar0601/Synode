/** Built-in datasets the API can load (services/api routes/datasets.py BUILTINS). */
export const BUILTINS = [
  { name: "adult", label: "Adult" },
  { name: "credit", label: "Credit default" },
  { name: "diabetes", label: "Diabetes 130-US" },
] as const;

export const isBuiltin = (id: string): boolean => BUILTINS.some((b) => b.name === id);
