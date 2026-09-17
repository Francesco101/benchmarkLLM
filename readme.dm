Adattando solamente __main__ benchmarks.py si eseguono i benchmark.

Per ora
  • Va strutturata un attimo la parte dei parametri passati a lm_eval, ma per ora per testare va bene.
  • gsm8k, mmlu sono praticamente saturati su Qwen3.6 quindi non dicono molto.
  • minerva_math ottine circa 90% accuracy, quindi è più interessante. (se non ci sono stati bug, domani controllo)
  • hendrics_math da vedere perché si buggava.
  • magari aggiungere altre categorie di test.
  • alcuni benchmark si trovano qui: https://epoch.ai/benchmarks?view=graph&tab=eci
