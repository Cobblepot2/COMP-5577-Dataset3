"""Check data leakage and reproducibility properties of the persisted split."""
import csv
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest


def labels_by_id(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows, {row["id"]: row["label"] for row in rows}


class SplitTests(unittest.TestCase):
    def test_partition_preserves_records_and_labels(self):
        source_rows, source = labels_by_id("data/dataset3/train.csv")
        train_rows, train = labels_by_id("splits/train.csv")
        val_rows, val = labels_by_id("splits/validation.csv")
        self.assertEqual(len(train_rows), len(train))
        self.assertEqual(len(val_rows), len(val))
        self.assertFalse(train.keys() & val.keys())
        self.assertEqual(train | val, source)
        self.assertEqual(len(train_rows) + len(val_rows), len(source_rows))
        self.assertEqual({r["label"] for r in train_rows}, {str(i) for i in range(10)})
        self.assertEqual({r["label"] for r in val_rows}, {str(i) for i in range(10)})

    def test_regeneration_matches_committed_split(self):
        paths = [Path("splits/train.csv"), Path("splits/validation.csv")]
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        subprocess.run([sys.executable, "scripts/prepare_dataset.py"], check=True,
                       stdout=subprocess.DEVNULL)
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])


if __name__ == "__main__":
    unittest.main()
