from pathlib import Path

from purchase_intent.data.loader import load_raw_data


def test_load_raw_data_reads_csv_into_dataframe(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("Administrative,Revenue\n0,FALSE\n1,TRUE\n")

    df = load_raw_data(csv_path)

    assert list(df.columns) == ["Administrative", "Revenue"]
    assert len(df) == 2
