from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuiltInPdf:
    path: Path
    company_name: str
    product_name: str

    @property
    def display_name(self) -> str:
        return f"{self.company_name}｜{self.product_name}"


def discover_builtin_pdfs(root: Path) -> tuple[BuiltInPdf, ...]:
    if not root.exists():
        return ()
    docs: list[BuiltInPdf] = []
    for path in sorted(root.rglob("*.pdf")):
        relative = path.relative_to(root)
        parts = relative.parts
        company = parts[0] if len(parts) >= 1 else "未知保险公司"
        product = parts[1] if len(parts) >= 2 else path.stem
        docs.append(BuiltInPdf(path=path, company_name=company, product_name=product))
    return tuple(docs)
