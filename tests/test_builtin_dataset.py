import shutil
import uuid
from pathlib import Path

from insurance_rag.builtin_dataset import discover_builtin_pdfs


def _unique_test_root() -> Path:
    return Path("tmp") / f"test_builtin_dataset_{uuid.uuid4().hex}"


def test_discover_builtin_pdfs_reads_nested_pdf_layout():
    root = _unique_test_root()
    try:
        pdf_path = root / "公司A" / "产品A" / "公司A_产品A_条款书.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4")

        docs = discover_builtin_pdfs(root)

        assert len(docs) == 1
        assert docs[0].company_name == "公司A"
        assert docs[0].product_name == "产品A"
        assert docs[0].path == pdf_path
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_discover_builtin_pdfs_returns_empty_tuple_for_missing_root():
    root = _unique_test_root()

    docs = discover_builtin_pdfs(root)

    assert docs == ()
