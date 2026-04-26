import csv
from math import log
from dataclasses import dataclass

SKIP_FIELDS = ['track_name']
MULTI_VALUE_FIELDS = ['artists']

@dataclass
class Document:
    identifier: str
    bag_of_attributes: list[str]
    term_frequencies: dict[str, int]
    tfidf_score: dict[str, int]


def load_dataset(filepath: str) -> list[dict]:
    with open(filepath, 'r') as file:
        return list(csv.DictReader(file))


def build_vocabulary(data: list[dict]) -> set[str]:
    vocabulary = set()
    for document in data:
        for field, value in document.items():
            if field in SKIP_FIELDS:
                continue
            if field in MULTI_VALUE_FIELDS:
                vocabulary.update(value.split(';'))
            else:
                vocabulary.add(value)

    return vocabulary


def build_corpus(data: list[dict], vocabulary: set[str]) -> list[Document]:
    corpus = []
    for document in data:
        bag_of_attributes = []
        term_frequencies = dict.fromkeys(vocabulary, 0)
        identifier = ''
        for field, value in document.items():
            if field == 'track_name':
                identifier = value
            elif field in MULTI_VALUE_FIELDS:
                bag_of_attributes.extend(value.split(';'))
            else:
                bag_of_attributes.append(value)

        tfidf_score = {}
        track = Document(identifier, bag_of_attributes, term_frequencies, tfidf_score)
        corpus.append(track)

    return corpus
            

def compute_document_frequencies(corpus: list[Document], vocabulary: set[str]) -> dict[str, int]:
    document_frequencies = dict.fromkeys(vocabulary, 0)
    for document in corpus:
        for term in set(document.bag_of_attributes):
            if term in vocabulary:
                document_frequencies[term] += 1

    return document_frequencies


def compute_TF(document: Document) -> dict[str, int]:
    tf_scores= {}
    for field, value in document.term_frequencies.items():
        tf_score = value / len(document.bag_of_attributes)
        tf_scores[field] = tf_score

    return tf_scores


def compute_IDF(total_number_documents: int, document_frequencies: dict[str, int]) -> dict[str, int]:
    IDF_scores = {}
    for field, value in document_frequencies.items():
        idf_score = log(total_number_documents / value)
        IDF_scores[field] = idf_score

    return IDF_scores


def tfidf(corpus: list[Document], document_frequencies: dict[str, int]):
    total_number_documents = len(corpus)
    IDF_scores = compute_IDF(total_number_documents, document_frequencies)


    for document in corpus:
        for word in document.bag_of_attributes:
            document.term_frequencies[word]+=1

        tf_scores = compute_TF(document)

        for field, value in tf_scores.items():
            tfidf_score = value * IDF_scores[field]
            document.tfidf_score[field] = tfidf_score


if __name__ == '__main__':
    data = load_dataset('test_dataset.csv')
    vocabulary = build_vocabulary(data)
    corpus = build_corpus(data, vocabulary)

    document_frequencies = compute_document_frequencies(corpus, vocabulary)
    tfidf(corpus, document_frequencies)

    for document in corpus:
        print(f"Track Name: {document.identifier}")
        print(f"TFIDF SCORE: {document.tfidf_score}")
        print()
        print()
        print()
        print()



