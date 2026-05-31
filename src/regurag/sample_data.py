from __future__ import annotations

from regurag.schemas import Chunk


def sample_chunks() -> list[Chunk]:
    """Tiny synthetic corpus used when real sources have not been ingested yet."""

    return [
        Chunk(
            chunk_id="0000000000000001",
            source_id="sample_ai_act",
            text=(
                "Synthetic sample. AI systems used for recruitment, candidate screening, "
                "or decisions affecting employment can require careful risk assessment "
                "under the EU AI Act. A production assistant must cite the exact official "
                "article before giving a compliance answer."
            ),
            metadata={
                "title": "Synthetic AI Act sample",
                "url": "local://sample_ai_act",
                "language": "en",
                "section_heading": "Employment and HR example",
            },
        ),
        Chunk(
            chunk_id="0000000000000002",
            source_id="sample_gdpr",
            text=(
                "Synthetic sample. GDPR analysis should consider personal data, legal "
                "basis, transparency, purpose limitation, data minimization, and data "
                "subject rights. Unsupported legal claims should be refused."
            ),
            metadata={
                "title": "Synthetic GDPR sample",
                "url": "local://sample_gdpr",
                "language": "en",
                "section_heading": "Privacy principles example",
            },
        ),
        Chunk(
            chunk_id="0000000000000003",
            source_id="sample_bnetza",
            text=(
                "Synthetisches Beispiel. Deutsche Unternehmen sollten KI-Kompetenz, "
                "Dokumentation, Risikoanalyse und Datenschutz zusammen betrachten, "
                "wenn sie KI-Systeme in regulierten Bereichen einsetzen."
            ),
            metadata={
                "title": "Synthetic Bundesnetzagentur sample",
                "url": "local://sample_bnetza",
                "language": "de",
                "section_heading": "KI-Kompetenz Beispiel",
            },
        ),
    ]
