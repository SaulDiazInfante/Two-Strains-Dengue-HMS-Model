# Reference Data

This directory stores the committed reference dataset used for local development and reproducible prep steps.

- Source data lives in `raw_data/`.
- The Python wheel does not bundle these files.
- Create a working copy for runs with:

```bash
two-strains-dengue prepare-data --output-dir ./artifacts/data
```

- Then run the model against that prepared directory with `--data-dir ./artifacts/data` or by setting `TWO_STRAINS_DENGUE_DATA_DIR`.
