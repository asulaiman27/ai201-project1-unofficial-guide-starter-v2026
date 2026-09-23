"""
Stage 2 of the pipeline: splitting documents into chunks.

⚠️ THIS IS THE FILE YOU CHANGE IN MILESTONE 3.

`split_documents` below is deliberately plain. It cuts every document into
fixed-size pieces with a fixed overlap and pays no attention to where sentences
or paragraphs end. It works, and it is not good.

On a corpus of short posts it may not cut anything at all: `campus_life` comes
out as 88 documents and 88 chunks, because almost nothing in it reaches 800
characters. That is the baseline, not a bug — Milestone 3 is where you decide
whether one post should stay one chunk.

Your job in Milestone 3 is to replace the *body* of `split_documents` with a
strategy that fits the documents you actually read in Milestone 1. Keep the
name and the shape of what it returns — the rest of the pipeline calls it, and
your README has to name the function that produced your chunks.

If you get stuck for 30 minutes, `fallback_split` is the original. Switch back
to it, write down what you saw, and move on. That's a real observation about
your pipeline, not giving up.
"""

from dataclasses import dataclass
import re

import config
from ingest import Document


@dataclass
class Chunk:
    """One piece of one document."""

    text: str
    source: str        # which file it came from
    index: int         # which chunk within that file, starting at 0
    produced_by: str   # the function that made it — cite this in your README

    @property
    def label(self) -> str:
        return f"{self.source}#{self.index}"


def fallback_split(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    The starter's original chunker. Fixed-size character windows with overlap.

    Keep this function. Milestone 3's stop rule points back at it, and having
    something to compare your own strategy against is useful in unit 2.
    """
    # Keep the original starter settings available for comparison even after
    # the custom chunker's settings in config.py change.
    chunk_size = 800 if chunk_size is None else chunk_size
    overlap = 120 if overlap is None else overlap

    if overlap >= chunk_size:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        start = 0
        index = 0
        while start < len(doc.text):
            piece = doc.text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::fallback_split",
                    )
                )
                index += 1
            start += chunk_size - overlap

    return chunks


def split_documents(documents: list[Document]) -> list[Chunk]:
    """
    Split the short campus posts on paragraph boundaries, up to a 400-char cap.

    A short first paragraph is treated as the post heading and repeated on each
    chunk. Paragraphs are packed together while they fit; an unusually long
    paragraph is split at sentence boundaries, with a word boundary as a last
    resort. No characters overlap between chunks.
    """
    limit = config.CHUNK_SIZE
    if limit < 1:
        raise ValueError("chunk size has to be positive")

    chunks: list[Chunk] = []
    for doc in documents:
        paragraphs = [
            re.sub(r"\s+", " ", part).strip()
            for part in re.split(r"\n\s*\n", doc.text)
            if part.strip()
        ]
        if not paragraphs:
            continue

        first = paragraphs[0]
        has_heading = (
            len(paragraphs) > 1
            and len(first) <= 100
            and not first.endswith((".", "!", "?"))
        )
        heading = first if has_heading else ""
        body_paragraphs = paragraphs[1:] if has_heading else paragraphs
        body_limit = limit - len(heading) - 2 if heading else limit
        if body_limit < 1:
            heading = ""
            body_limit = limit
            body_paragraphs = paragraphs

        def split_long_paragraph(paragraph: str) -> list[str]:
            if len(paragraph) <= body_limit:
                return [paragraph]
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            pieces: list[str] = []
            current = ""
            for sentence in sentences:
                # If a sentence itself is too long, retain whole words where
                # possible and split only that sentence into smaller pieces.
                words = [sentence[i:i + body_limit] for i in range(0, len(sentence), body_limit)] \
                    if not re.search(r"\s", sentence) else sentence.split()
                sentence_pieces: list[str] = []
                word_piece = ""
                for word in words:
                    if len(word) > body_limit:
                        if word_piece:
                            sentence_pieces.append(word_piece)
                            word_piece = ""
                        sentence_pieces.extend(
                            word[i:i + body_limit] for i in range(0, len(word), body_limit)
                        )
                    elif word_piece and len(word_piece) + 1 + len(word) > body_limit:
                        sentence_pieces.append(word_piece)
                        word_piece = word
                    else:
                        word_piece = f"{word_piece} {word}".strip()
                if word_piece:
                    sentence_pieces.append(word_piece)

                for piece in sentence_pieces:
                    candidate = f"{current} {piece}".strip()
                    if current and len(candidate) > body_limit:
                        pieces.append(current)
                        current = piece
                    else:
                        current = candidate
            if current:
                pieces.append(current)
            return pieces

        body_pieces = [
            piece
            for paragraph in body_paragraphs
            for piece in split_long_paragraph(paragraph)
        ]

        grouped: list[str] = []
        current_parts: list[str] = []
        for piece in body_pieces:
            candidate = "\n\n".join(current_parts + [piece])
            if current_parts and len(candidate) > body_limit:
                grouped.append("\n\n".join(current_parts))
                current_parts = [piece]
            else:
                current_parts.append(piece)
        if current_parts:
            grouped.append("\n\n".join(current_parts))

        for index, body in enumerate(grouped):
            text = f"{heading}\n\n{body}" if heading else body
            chunks.append(
                Chunk(
                    text=text,
                    source=doc.source,
                    index=index,
                    produced_by="chunker.py::split_documents",
                )
            )

    return chunks


def describe(chunks: list[Chunk]) -> str:
    """A one-line summary, printed after indexing."""
    if not chunks:
        return "0 chunks"
    lengths = [len(c.text) for c in chunks]
    return (
        f"{len(chunks)} chunks, "
        f"{sum(lengths) // len(lengths)} characters on average "
        f"(shortest {min(lengths)}, longest {max(lengths)}), "
        f"produced by {chunks[0].produced_by}"
    )


if __name__ == "__main__":
    from ingest import load_documents

    chunks = split_documents(load_documents())
    print(describe(chunks))
