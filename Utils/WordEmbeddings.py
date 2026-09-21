import os
import hashlib
import fasttext
import numpy as np
from tqdm import tqdm
from Utils.Identifier import Identifier as id


def _load_text_vectors(filepath):
    """
    Load embeddings from a text-format file (GloVe .txt, FastText .vec, IndicNLP .vec, MuRIL .vec).
    Each line: word dim1 dim2 ... dimN
    First line of FastText/IndicNLP files is a header (num_words dim) -- detected and skipped.
    Returns a dict: {word_string: numpy_array}.
    """
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        # Heuristic: if line has exactly 2 tokens and both are digits, it is a header
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            pass  # skip header
        else:
            # First line is actual data (GloVe format has no header)
            word = parts[0]
            try:
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[word] = vector
            except ValueError:
                pass

        for line in tqdm(f, desc="LOADING PRETRAINED EMBEDDINGS"):
            parts = line.rstrip().split(' ')
            word = parts[0]
            try:
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[word] = vector
            except ValueError:
                continue  # skip malformed lines
    return embeddings


def _load_word2vec_binary(filepath):
    """
    Load Word2Vec binary format (.bin) using gensim.
    Returns a dict: {word_string: numpy_array}.
    """
    from gensim.models import KeyedVectors
    print(">>> Loading Word2Vec binary format (this may take a while)...")
    model = KeyedVectors.load_word2vec_format(filepath, binary=True)
    embeddings = {}
    for word in tqdm(model.key_to_index, desc="EXTRACTING WORD2VEC VECTORS"):
        embeddings[word] = model[word]
    return embeddings


def _load_fasttext_binary(filepath):
    """
    Load FastText binary format (.bin) using the fasttext library.
    Returns a dict: {word_string: numpy_array}.
    """
    print(">>> Loading FastText binary format (this may take a while)...")
    model = fasttext.load_model(filepath)
    embeddings = {}
    for word in tqdm(model.get_words(), desc="EXTRACTING FASTTEXT VECTORS"):
        embeddings[word] = model.get_word_vector(word)
    return embeddings


def _get_loader(embedding_type):
    """Return the appropriate loader function for the given embedding type."""
    if embedding_type == id.embedding_word2vec_googlenews:
        return _load_word2vec_binary
    elif embedding_type == id.embedding_fasttext_hindi:
        return _load_fasttext_binary
    else:
        # GloVe, IndicNLP, MuRIL all use text format
        return _load_text_vectors


def _get_cache_path(project_root, embedding_type, vocabulary):
    """Return the path for the cached vocabulary-specific embedding matrix."""
    cache_dir = os.path.join(project_root, id.resource_saving_path, "CustomizedPretrainedEmbeddings")
    os.makedirs(cache_dir, exist_ok=True)
    vocab_text = "\n".join(vocabulary).encode("utf-8")
    vocab_hash = hashlib.sha1(vocab_text).hexdigest()[:10]
    return os.path.join(cache_dir, f"{embedding_type}_vocab{len(vocabulary)}_{vocab_hash}.npy")


def build_embedding_matrix(vocabulary, embedding_type, project_root):
    """
    Build a pretrained embedding matrix aligned with TextVectorization indexing.
    Uses a cached .npy file if available, otherwise loads the full pretrained file
    once, extracts only the vocabulary words, and caches the result for future use.

    TextVectorization index mapping:
        0 -> padding token  (must remain zero vector for mask_zero=True)
        1 -> OOV token      (random initialization)
        2 -> vocabulary[0]
        3 -> vocabulary[1]
        ...
        N+1 -> vocabulary[N-1]

    Args:
        vocabulary: list of strings -- the vocabulary list from Dataset.vocabulary
        embedding_type: one of the Identifier embedding constants (string)
        project_root: absolute path to the project root directory

    Returns:
        embedding_matrix: numpy array of shape (len(vocabulary) + 2, embedding_dim)
        embedding_dim: int, the dimensionality of the pretrained embeddings
    """
    embedding_dim = id.pretrained_embedding_dims[embedding_type]
    cache_path = _get_cache_path(project_root, embedding_type, vocabulary)

    # Check if a cached vocabulary-specific matrix already exists
    if os.path.exists(cache_path):
        embedding_matrix = np.load(cache_path)
        print(f">>> Loaded cached embedding matrix from {cache_path}")
        print(f">>> Matrix shape: {embedding_matrix.shape} ({embedding_matrix.nbytes / 1024:.1f} KB)")
        return embedding_matrix, embedding_dim

    # No cache found — load the full pretrained file
    relative_path = id.pretrained_embedding_files[embedding_type]
    filepath = os.path.join(project_root, relative_path)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Pretrained embedding file not found: {filepath}\n"
            f"Please download the {embedding_type} embeddings and place them at this path."
        )

    loader = _get_loader(embedding_type)
    print(f">>> Loading pretrained embeddings: {embedding_type} from {filepath}")
    pretrained_vectors = loader(filepath)
    print(f">>> Loaded {len(pretrained_vectors)} pretrained word vectors")

    # Build the matrix: vocab_size + 2 rows (pad + OOV + vocab words)
    matrix_size = len(vocabulary) + 2
    embedding_matrix = np.zeros((matrix_size, embedding_dim), dtype=np.float32)

    # Row 0 = padding -> stays all zeros (required by mask_zero=True)
    # Row 1 = OOV token -> random initialization
    embedding_matrix[1] = np.random.uniform(-0.05, 0.05, size=embedding_dim)

    # Rows 2..N+1 = vocabulary words
    found_count = 0
    missing_words = []
    for i, word in enumerate(vocabulary):
        if word in pretrained_vectors:
            vec = pretrained_vectors[word]
            # Guard against dimension mismatch from corrupted lines
            if len(vec) == embedding_dim:
                embedding_matrix[i + 2] = vec
                found_count += 1
            else:
                embedding_matrix[i + 2] = np.random.uniform(-0.05, 0.05, size=embedding_dim)
                missing_words.append(word)
        else:
            # Word not in pretrained vocabulary -> random initialization
            embedding_matrix[i + 2] = np.random.uniform(-0.05, 0.05, size=embedding_dim)
            missing_words.append(word)

    coverage = found_count / len(vocabulary) * 100
    print(f">>> Embedding coverage: {found_count}/{len(vocabulary)} words ({coverage:.1f}%)")
    print(f">>> {len(missing_words)} words not found in pretrained embeddings (randomly initialized)")

    # Free memory from the large pretrained dict
    del pretrained_vectors

    # Cache the small matrix for future runs
    np.save(cache_path, embedding_matrix)
    print(f">>> Cached embedding matrix to {cache_path} ({embedding_matrix.nbytes / 1024:.1f} KB)")

    return embedding_matrix, embedding_dim
