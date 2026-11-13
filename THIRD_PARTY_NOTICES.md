# Third-party materials

The MIT license covers the original source code and documentation contributed
to this repository. It does not grant rights in independently licensed datasets,
pretrained models, dependencies, or third-party implementations.

- **Amazon Reviews 2014 / 5-core** reviews and source metadata must be obtained
  from their original providers and used under their terms. Raw reviews,
  item-level metadata, IDs, and extracted Title embedding arrays are not part of
  the public summary bundle.
- **all-MiniLM-L6-v2** is loaded through Sentence Transformers for frozen Title
  representations. Obtain the model separately and follow its model-card license.
- **SASRec-based implementations** in this repository are the implementations
  evaluated in this study. The repository does not claim to be an exact
  reproduction of SASRecF, FDSA, NOVA, DIF-SR, DIFF, or other cited architectures.
- Python dependencies retain their own licenses. Installing this package does
  not replace those licenses with MIT.

The files in `results/paper/` are aggregate experimental summaries and provenance
records. They contain no model weights, full review text, or per-user prediction
arrays. Historical local data and model artifacts are outside this source release.
