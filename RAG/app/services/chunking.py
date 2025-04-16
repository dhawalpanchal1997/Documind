import spacy
from typing import List, Tuple
import re

class SpacyChunker:
    def __init__(self, model_name: str = "en_core_web_sm", max_chunk_size: int = 5000):
        self.nlp = spacy.load(model_name)
        self.max_chunk_size = max_chunk_size

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def create_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, str]]:
        text = self.clean_text(text)
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        chunks = []
        current_chunk = ""
        chunk_count = 1
        i = 0

        while i < len(sentences):
            sentence = sentences[i]
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
                i += 1
            else:
                if current_chunk:
                    chunks.append((chunk_count, current_chunk))
                    chunk_count += 1
                    # Add overlap to next chunk
                    overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap < len(current_chunk) else current_chunk
                    current_chunk = overlap_text
                else:
                    chunks.append((chunk_count, sentence[:chunk_size]))
                    chunk_count += 1
                    i += 1

        if current_chunk:
            chunks.append((chunk_count, current_chunk))

        return chunks

def chunk_document(document: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, str]]:
    chunker = SpacyChunker()
    return chunker.create_chunks(document, chunk_size, chunk_overlap)
