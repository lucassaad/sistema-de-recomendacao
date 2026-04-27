import csv, random
from math import log
from dataclasses import dataclass

def generate_random_matrix(rows: int, cols: int, vals: range, seed: int = 1) -> list:
	random.seed(seed)
	return [[random.choice(vals) for _ in range(cols)] for _ in range(rows)]

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


def build_user_profile(user_ratings: list[int], corpus: list[Document]) -> dict[str, float]:
    user_profile = {}
    
    for rating, document in zip(user_ratings, corpus):
        if rating:
            for word, tfidf_val in document.tfidf_score.items():
                weighted_score = tfidf_val * rating
                user_profile[word] = user_profile.get(word, 0.0) + weighted_score
                
    return user_profile


def cosine_similarity(profile_a: dict[str, float], profile_b: dict[str, float]) -> float:
    common_words = set(profile_a.keys()).intersection(set(profile_b.keys()))
    dot_product = sum(profile_a[word] * profile_b[word] for word in common_words)
    
    mag_a = sum(val ** 2 for val in profile_a.values()) ** 0.5
    mag_b = sum(val ** 2 for val in profile_b.values()) ** 0.5
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
        
    return dot_product / (mag_a * mag_b)


def register_new_user(corpus: list, num_samples: int = 5) -> list[int]:
    print("Para iniciar, avalie as seguintes musicas com valores inteiros de 1 a 5, sendo 1 o pior e 5 o melhor nivel de avaliacao.")
    print("(Insira 0 se nao conhecer ou nao quiser avaliar a musica)\n")
    
    user_ratings = [0] * len(corpus)
    
    sample_indices = random.sample(range(len(corpus)), num_samples)
    
    for idx in sample_indices:
        song = corpus[idx]
        while True:
            try:
                rating = int(input(f"Como voce avalia '{song.identifier}'? (0-5): "))
                if 0 <= rating <= 5:
                    user_ratings[idx] = rating
                    break
                else:
                    print("Insira um numero valido entre 0 e 5.")
            except ValueError:
                print("Entrada invalida. Por favor, insira um inteiro.")
                
    print("\nGerando perfil personalizado...\n")
    return user_ratings


def recommend_songs(user_ratings: list[int], corpus: list, top_n: int = 5) -> list[tuple]:
    user_profile = build_user_profile(user_ratings, corpus)
    recommendations = []
    
    for i, document in enumerate(corpus):
        if not user_ratings[i]:
            sim_score = cosine_similarity(user_profile, document.tfidf_score)
            recommendations.append((document.identifier, sim_score))
            
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations[:top_n]


def setup_recommender(filepath: str, num_users: int = 100, seed: int = 42):
    print("Inicializando Sistema de Recomendacao...")
    
    data = load_dataset(filepath)
    vocabulary = build_vocabulary(data)
    corpus = build_corpus(data, vocabulary)
    
    document_frequencies = compute_document_frequencies(corpus, vocabulary)
    tfidf(corpus, document_frequencies)
    
    num_songs = len(corpus)
    utility_matrix = generate_random_matrix(rows=num_users, cols=num_songs, vals=range(0, 6), seed=seed)
    
    print(f"Inicializacao completa! {num_songs} musicas foram adicionadas e uma matriz de utilidade {num_users}x{num_songs} foi gerada.")
    
    return corpus, utility_matrix


def run_recommender_ui(corpus: list):
    print("="*40)
    print("--- Sistema de Recomendacao de Musicas Nacionais - IIA ---")
    print("="*40)
    
    new_user_ratings = register_new_user(corpus, num_samples=5)
    
    top_recommendations = recommend_songs(new_user_ratings, corpus, top_n=5)
    
    print("\n" + "="*40)
    if top_recommendations:
        print("     Recomendacoes Personalizadas para Voce:     ")
        print("="*40)
        for rank, (song_name, score) in enumerate(top_recommendations, start=1):
            print(f"{rank}. {song_name} (Score: {score:.4f})")
    else:
        print("Nao foi possivel gerar recomendacoes. Por favor, avalie mais musicas para melhorar seu perfil.")
    print("="*40)


if __name__ == '__main__':
    corpus, utility_matrix = setup_recommender('dataset.csv')
    run_recommender_ui(corpus)

'''
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
'''
