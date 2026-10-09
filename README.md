# SmartLink updates

Copy the `src/`, `scripts/`, and `tests/` folders into your existing SmartLink project root, preserving relative paths. Replace the existing files listed in `FILES.json`; add the three new files. This archive is an overlay, not a complete repository. Keep a copy of the existing files when reviewing the changes.

## Scope

This connects the existing driver to daily grouping, candidate generation, validation, and candidate history. It preserves the existing ITRON assignment algorithm and BallTree/haversine search. Candidate batches have atomic files and completion manifests; meters with no candidates have an explicit ledger. Candidate history includes removed entities, removed divisions, grouping/context changes, and surviving/removed pairs.

Candidate-feature history, incremental spatial/tile actions, training datasets, and model training are subsequent work. This update does not implement COMPUTE/RECOMPUTE/DROP/REUSE for tile features. Existing feature, tile, training, meter-to-meter, and legacy source files remain unchanged.

## Environment

`src/mtx/db/db_AMI.py` now reads its password from `SMARTLINK_AMI_PASSWORD`. Set that environment variable before running a database-connected pipeline. `SMARTLINK_AMI_USERNAME` is optional and defaults to the existing service username. No password is included in this archive.

For a short Windows output path, set `SMARTLINK_M2T_DIRECTORY` to a directory such as `C:\sl_m2t`. The default is `<project>/data/processed/m2t`. Transformer and ADAM source cache overrides are available through `SMARTLINK_TRANSFORMER_CACHE_DIRECTORY` and `SMARTLINK_ADAM_CACHE_DIRECTORY`.

Existing raw source cache/history locations remain available. Source histories compare complete standardized extracts before coordinate rejection. Final grouping history has its own root so it cannot mix schemas with raw ITRON history. The new candidate-history root uses its own manifest layout; older notebook-produced candidate history is not silently migrated.

## Daily behavior

A complete cache is reused when its date, input/configuration signature, and files agree. A new date builds grouping/candidates and compares the result with the previous committed candidate state. Completed candidate batches can be reused after interruption. Empty batches are recorded explicitly.

The first implementation deliberately builds candidates for the new day's eligible division population. It records the pair changes needed for later incremental feature/tile integration; it does not yet skip unchanged candidate searches across dates.

A started daily run pins its inputs/configuration. Existing source histories freeze a committed date, so newer same-day data/configuration is rejected instead of mixing it with saved source-history results. `overwrite_m2t_outputs` can rebuild unchanged daily stage outputs; it does not amend committed history. Run one writer at a time. The `main` result is now a dictionary exposing grouping/candidate paths, zero-candidate ledgers, candidate-summary and pair-change paths, and the source-history/ADAM training-label events.

## Validation

From the repository root, run:

```console
python -m unittest discover -s tests -p "test_m2t_daily_contract.py" -v
```

Nine focused checks cover radius/parity, duplicate and division guards, zero-candidate conservation, group preservation, verified-label conflicts, calendar-day ITRON reuse, daily caches, candidate batch crash/restart, candidate history removals, immutable reuse, and commit recovery.

Validation performed here: changed Python files compiled; six in-memory/calendar checks passed. Three real-Parquet filesystem checks were skipped because pyarrow is unavailable in this environment. Their restart/cache/history protocols also passed with a temporary pickle substitute; that does not validate the Parquet engine. GIS/CRS, Windows rename behavior, and the full database-connected driver have not been executed here. Run the supplied checks with your installed project dependencies, then run a representative division and verify the real outputs before a full production run. No production data, database writes, existing repository files, or existing histories were modified while preparing this archive.
