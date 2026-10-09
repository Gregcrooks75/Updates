"""Focused contracts for daily grouping -> candidates -> candidate history.

Run from the repository root:
    python -m unittest discover -s tests -p "test_m2t_daily_contract.py" -v
Filesystem tests use real Parquet when pyarrow is installed (a project dependency).
No database, GIS files, or production history is touched.
"""
from __future__ import annotations
import importlib.util
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from mtx.run_config import RunConfig
from mtx.grouping.group_assignments import build_itron_group_assignments, GROUP_COLUMNS
from mtx.candidates import transformer_candidates as candidates
from mtx.history.candidate_history import compare_candidate_pairs, track_candidate_history
from mtx.utils.daily_cache import read_json
from meter_to_transformer.stages.run_meter_grouping import run_meter_grouping_stage

HAS_PARQUET = importlib.util.find_spec("pyarrow") is not None


def config(**overrides):
    values = dict(week_start="2026-02-02", week_end="2026-02-08",
                  correlation_start="2026-02-02", correlation_end="2026-02-08",
                  transformer_division_col="Division", transformer_candidate_batch_size=1)
    values.update(overrides)
    return RunConfig(**values)


def inputs():
    ami = pd.DataFrame({"meter_number": ["M1", "M2"], "Division": ["Central"] * 2,
                        "latitude": [43.0, 43.1], "longitude": [-77.0, -77.0],
                        "current_meter_floc": ["F1", "F2"]})
    itron = pd.DataFrame({"meter_number": ["M1", "M2"], "Division": ["Central"] * 2,
                          "Tx_Subgroup_ID": ["G1", "G1"]})
    tx = pd.DataFrame({"transformer_floc": ["T1", "T2"], "Division": ["Central"] * 2,
                       "latitude": [43.0002, 43.0005], "longitude": [-77.0, -77.0],
                       "sap_id": ["S1", "S2"], "gis_id": ["1", "2"]})
    groups = build_itron_group_assignments(ami[["meter_number", "Division"]], itron)
    return ami, itron, tx, groups


def zero_ledger(groups, pairs):
    zero = groups.loc[~groups["meter_number"].isin(pairs["meter_number"])].copy()
    zero["zero_candidate_reason"] = "NO_TRANSFORMER_WITHIN_RADIUS"
    return zero


class InMemoryContracts(unittest.TestCase):
    def test_standardized_and_legacy_candidates_match_brute_force(self):
        ami, _, tx, groups = inputs()
        actual = candidates.find_transformer_candidates(groups, ami, tx, "Central")
        expected = {}
        for meter in ami.itertuples():
            for transformer in tx.itertuples():
                lat1, lat2 = np.radians([meter.latitude, transformer.latitude])
                dlat, dlon = lat2 - lat1, np.radians(transformer.longitude - meter.longitude)
                feet = 2 * np.arcsin(np.sqrt(np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)) * candidates.EARTH_RADIUS_FT
                if feet <= 550:
                    expected[(meter.meter_number, transformer.transformer_floc)] = feet
        self.assertEqual(set(zip(actual.meter_number, actual.candidate_transformer)), set(expected))
        for row in actual.itertuples():
            self.assertAlmostEqual(row.distance_ft, expected[(row.meter_number, row.candidate_transformer)], places=6)
        legacy_ami = ami.rename(columns={"latitude": "LATITUDE", "longitude": "LONGITUDE", "meter_number": "MeterID"})
        legacy_tx = tx.rename(columns={"transformer_floc": "FuncLoc", "latitude": "Location_Y", "longitude": "Location_X", "Division": "DivName"})
        legacy_tx["DivName"] = "Central/Rochester"
        legacy = candidates.find_transformer_candidates(groups, legacy_ami, legacy_tx, "Central", transformer_division_col="DivName")
        pd.testing.assert_frame_equal(actual, legacy)
        candidates.validate_candidate_frames(actual, groups, zero_ledger(groups, actual),
                                             division="Central", radius_ft=550, transformer_df=tx)

    def test_validator_rejects_duplicates_loss_bad_distances_and_cross_division(self):
        ami, _, tx, groups = inputs()
        good = candidates.find_transformer_candidates(groups, ami, tx, "Central")
        zero = zero_ledger(groups, good)
        bad_distance = good.copy(); bad_distance.loc[0, "distance_ft"] = np.nan
        too_far = good.copy(); too_far.loc[0, "distance_ft"] = 551
        wrong_division = good.copy(); wrong_division.loc[0, "division"] = "Auburn"
        for bad, ledger in [(pd.concat([good, good.iloc[:1]], ignore_index=True), zero),
                            (bad_distance, zero), (too_far, zero), (wrong_division, zero),
                            (good, zero.iloc[:0])]:
            with self.subTest(rows=len(bad)), self.assertRaises(AssertionError):
                candidates.validate_candidate_frames(bad, groups, ledger, division="Central", radius_ft=550, transformer_df=tx)

    def test_pair_history_tracks_all_four_states_and_context_changes(self):
        previous = pd.DataFrame({"meter_number": ["M1", "M2", "M3"], "candidate_transformer": ["T1"] * 3,
                                 "division": ["Central"] * 3, "distance_ft": [10.0] * 3,
                                 "meter_latitude": [43.0] * 3, "tx_group_id": ["G1"] * 3})
        current = previous.loc[previous["meter_number"].ne("M2")].copy()
        current.loc[current["meter_number"].eq("M3"), "meter_latitude"] = 43.001
        current.loc[current["meter_number"].eq("M3"), "tx_group_id"] = "G2"
        current = pd.concat([current, previous.iloc[:1].assign(meter_number="M4")], ignore_index=True)
        changes = compare_candidate_pairs(current, previous).set_index("meter_number")
        self.assertEqual(changes["change_type"].to_dict(), {"M1": "UNCHANGED", "M2": "REMOVED", "M3": "MODIFIED", "M4": "NEW"})
        self.assertTrue(changes.loc["M3", "coordinates_changed"])
        self.assertTrue(changes.loc["M3", "grouping_changed"])
        self.assertFalse(changes.loc["M2", "current_exists"])
        with self.assertRaises(ValueError):
            compare_candidate_pairs(pd.concat([current, current.iloc[:1]]), previous)

    def test_grouping_retains_standalones_and_candidate_inputs_reject_conflicts(self):
        ami, itron, tx, _ = inputs()
        groups = build_itron_group_assignments(ami, itron.iloc[:1])
        self.assertEqual(set(groups["meter_number"]), {"M1", "M2"})
        self.assertEqual(groups.set_index("meter_number").loc["M2", "meter_type"], "standalone")
        conflicting_tx = pd.concat([tx, tx.iloc[:1].assign(latitude=44.0)], ignore_index=True)
        with self.assertRaises(ValueError):
            candidates.find_transformer_candidates(groups, ami, conflicting_tx, "Central")
        with self.assertRaises(ValueError):
            candidates.find_transformer_candidates(groups, pd.concat([ami, ami.iloc[:1]]), tx, "Central")
        with self.assertRaises(ValueError):
            build_itron_group_assignments(ami, pd.concat([itron, itron.iloc[:1].assign(Tx_Subgroup_ID="OTHER")]))

    def test_itron_calendar_day_reuse_does_not_use_yesterdays_cache(self):
        from mtx.ingestion.itron_group_cache import cache_is_fresh
        with tempfile.TemporaryDirectory() as directory:
            cache_file, metadata_file = Path(directory) / "cache.parquet", Path(directory) / "metadata.json"
            cache_file.write_bytes(b"placeholder")
            metadata_file.write_text(json.dumps({"created_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                                                "cache_date": (date.today() - timedelta(days=1)).isoformat()}))
            self.assertTrue(cache_is_fresh(cache_file=cache_file, metadata_file=metadata_file))
            self.assertFalse(cache_is_fresh(cache_file=cache_file, metadata_file=metadata_file, cache_day=date.today()))

    def test_adam_conflicting_labels_are_not_silently_selected(self):
        fake_db = types.ModuleType("mtx.db.db_AMI")
        fake_db.get_adam_validated_data = lambda: None
        with patch.dict(sys.modules, {"mtx.db.db_AMI": fake_db}):
            from mtx.ingestion.adam_cache import validate_adam
        bad = pd.DataFrame({"meter_number": ["M1", "M1"], "Correct_Transformer_FLOC": ["T1", "T2"]})
        with self.assertRaises(ValueError):
            validate_adam(bad)
        valid = validate_adam(pd.DataFrame({"meter_number": [" M1 "], "Correct_Transformer_FLOC": [None]}))
        self.assertEqual(valid.loc[0, "meter_number"], "M1")
        self.assertTrue(pd.isna(valid.loc[0, "Correct_Transformer_FLOC"]))


@unittest.skipUnless(HAS_PARQUET, "Install project dependency pyarrow to run filesystem contracts.")
class FilesystemContracts(unittest.TestCase):
    def test_daily_group_cache_reuse_new_day_and_corruption(self):
        ami, itron, _, _ = inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_meter_grouping_stage(config(), ami, itron, snapshot_date="2026-03-01", output_dir=root)
            with patch("meter_to_transformer.stages.run_meter_grouping.build_itron_group_assignments", side_effect=AssertionError("Must reuse")):
                second = run_meter_grouping_stage(config(), ami, itron, snapshot_date="2026-03-01", output_dir=root)
            self.assertEqual(first, second)
            changed = itron.assign(Tx_Subgroup_ID="G2")
            with self.assertRaises(ValueError):
                run_meter_grouping_stage(config(), ami, changed, snapshot_date="2026-03-01", output_dir=root)
            next_day = run_meter_grouping_stage(config(), ami, changed, snapshot_date="2026-03-02", output_dir=root)
            self.assertNotEqual(first, next_day)
            first["Central"].write_bytes(b"interrupted parquet")
            repaired = run_meter_grouping_stage(config(), ami, itron, snapshot_date="2026-03-01", output_dir=root)
            self.assertEqual(len(pd.read_parquet(repaired["Central"])), 2)

    def test_candidate_batch_crash_restart_and_same_day_reuse(self):
        ami, _, tx, groups = inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = candidates._batch_candidates
            count = 0
            def fail_second(*args):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError("Injected crash")
                return original(*args)
            with patch.object(candidates, "_batch_candidates", side_effect=fail_second):
                with self.assertRaises(RuntimeError):
                    candidates.build_transformer_candidate_partitions(groups, ami, tx, division="Central", output_dir=root, batch_size=1)
            self.assertTrue((root / "b00000.json").exists())
            self.assertFalse((root / "manifest.json").exists())
            with patch.object(candidates, "_batch_candidates", wraps=original) as query:
                output = candidates.build_transformer_candidate_partitions(groups, ami, tx, division="Central", output_dir=root, batch_size=1)
                self.assertEqual(query.call_count, 1)
            self.assertTrue(read_json(root / "manifest.json")["complete"])
            self.assertEqual(len(pd.read_parquet(output)), 2)
            self.assertEqual(pd.read_parquet(root / "zero.parquet")["meter_number"].tolist(), ["M2"])
            with patch.object(candidates, "_batch_candidates", side_effect=AssertionError("Must reuse")):
                self.assertEqual(candidates.build_transformer_candidate_partitions(groups, ami, tx, division="Central", output_dir=root, batch_size=1), output)
            output.write_bytes(b"interrupted final file")
            orphan = root / ".w123456789abc.tmp"
            orphan.write_bytes(b"interrupted temporary write")
            with patch.object(candidates, "_batch_candidates", side_effect=AssertionError("Completed batches must survive")):
                repaired = candidates.build_transformer_candidate_partitions(groups, ami, tx, division="Central", output_dir=root, batch_size=1)
            self.assertEqual(len(pd.read_parquet(repaired)), 2)
            self.assertFalse(orphan.exists())

    def test_history_removals_zero_population_reuse_and_recovery(self):
        ami, itron, tx, groups = inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def stage(day, meters, transformers, assignments):
                output = candidates.build_transformer_candidate_partitions(assignments, meters, transformers, division="Central", output_dir=root / "candidates" / day, batch_size=1)
                return {"Central": output}
            files = stage("2026-03-01", ami, tx, groups)
            history_root = root / "history"
            summary = track_candidate_history(files, ami, groups, tx, snapshot_date="2026-03-01", history_dir=history_root)
            self.assertEqual(int(summary.loc[0, "meters_without_candidates"]), 1)
            # Simulate a crash after the daily manifest, before pointer promotion.
            (history_root / "latest.json").unlink()
            again = track_candidate_history(files, ami, groups, tx, snapshot_date="2026-03-01", history_dir=history_root)
            pd.testing.assert_frame_equal(summary, again)
            newer_ami = pd.concat([ami.iloc[:1], ami.iloc[:1].assign(meter_number="M3")], ignore_index=True)
            newer_groups = build_itron_group_assignments(newer_ami, itron.iloc[:1])
            newer_tx = tx.iloc[:1]
            newer_files = stage("2026-03-02", newer_ami, newer_tx, newer_groups)
            track_candidate_history(newer_files, newer_ami, newer_groups, newer_tx, snapshot_date="2026-03-02", history_dir=history_root)
            pairs = pd.read_parquet(history_root / "pair_changes" / "2026-03-02" / "Central.parquet")
            self.assertTrue(pairs["change_type"].eq("REMOVED").any())
            self.assertTrue(pairs["change_type"].eq("NEW").any())
            status = pd.read_parquet(history_root / "meter_status" / "2026-03-02" / "Central.parquet")
            self.assertEqual(status.set_index("meter_number").loc["M2", "change_type"], "REMOVED")
            tx_status = pd.read_parquet(history_root / "transformer_status" / "2026-03-02" / "Central.parquet")
            self.assertEqual(tx_status.set_index("transformer_floc").loc["T2", "change_type"], "REMOVED")
            with self.assertRaises(ValueError):
                track_candidate_history(newer_files, newer_ami.assign(current_meter_floc="DIFFERENT"), newer_groups, newer_tx,
                                        snapshot_date="2026-03-02", history_dir=history_root)
            unchanged_root = root / "history_unchanged"
            track_candidate_history(files, ami, groups, tx, snapshot_date="2026-03-01", history_dir=unchanged_root)
            unchanged_summary = track_candidate_history(files, ami, groups, tx, snapshot_date="2026-03-02", history_dir=unchanged_root)
            self.assertEqual(int(unchanged_summary.loc[0, "modified_pairs"]), 0)
            self.assertEqual(int(unchanged_summary.loc[0, "unchanged_pairs"]), 2)
            # A manifest can survive before either convenience output exists.
            recovery_root = root / "history_recovery"
            with patch("mtx.history.candidate_history._finish_commit", side_effect=RuntimeError("Injected pointer crash")):
                with self.assertRaises(RuntimeError):
                    track_candidate_history(files, ami, groups, tx, snapshot_date="2026-03-01", history_dir=recovery_root)
            self.assertFalse((recovery_root / "latest.json").exists())
            self.assertFalse((recovery_root / "candidate_daily_summary.parquet").exists())
            recovered_summary = track_candidate_history(newer_files, newer_ami, newer_groups, newer_tx,
                                                        snapshot_date="2026-03-02", history_dir=recovery_root)
            self.assertGreater(int(recovered_summary.loc[0, "removed_pairs"]), 0)
            # Entire division disappears; old pair/entity tombstones are retained.
            track_candidate_history({}, newer_ami.iloc[:0], newer_groups.iloc[:0], newer_tx.iloc[:0],
                                    snapshot_date="2026-03-03", history_dir=history_root)
            removed = pd.read_parquet(history_root / "pair_changes" / "2026-03-03" / "Central.parquet")
            self.assertTrue(removed["change_type"].eq("REMOVED").all())
            empty_meters = pd.read_parquet(history_root / "zero_candidate_meters" / "2026-03-03" / "Central.parquet")
            self.assertEqual(len(empty_meters), 0)


if __name__ == "__main__":
    unittest.main()
