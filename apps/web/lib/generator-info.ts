/** One plain-English sentence per catalog generator (mirrors packages/core/src/stl_core/catalog.py). */
export const GENERATOR_INFO: Record<string, string> = {
  train_copy: "Returns real training rows — a privacy sanity check that must fail the gate.",
  independent_marginals: "Samples each column independently from its own distribution, ignoring all relationships.",
  marginals: "Synthcity's marginal-distribution baseline: per-column sampling with no cross-column structure.",
  gaussian_copula: "Models each column's distribution plus pairwise correlations with a Gaussian copula.",
  bayesian_network: "Learns a graph of column dependencies and samples rows from it.",
  ctgan: "A conditional GAN for tabular data; handles mixed numeric and categorical columns.",
  tvae: "A variational autoencoder for tabular data; often a strong, stable deep baseline.",
  tabddpm: "A diffusion model for tabular data; high fidelity but needs a GPU.",
  great: "Fine-tunes a language model on rows rendered as text; slow and needs a GPU.",
  "mst_eps0.1": "Differentially private (epsilon 0.1, strongest privacy): MST synthesises from noisy marginals.",
  mst_eps1: "Differentially private (epsilon 1): MST synthesises from noisy marginals.",
  mst_eps10: "Differentially private (epsilon 10, weakest privacy): MST synthesises from noisy marginals.",
  "privbayes_eps0.1": "Audit only: synthcity PrivBayes is not actually differentially private (see report).",
  privbayes_eps1: "Audit only: synthcity PrivBayes is not actually differentially private (see report).",
  privbayes_eps10: "Audit only: synthcity PrivBayes is not actually differentially private (see report).",
};

export function generatorDescription(name: string): string {
  return GENERATOR_INFO[name] ?? "No description available.";
}
