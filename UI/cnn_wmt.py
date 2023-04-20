import pickle
import tensorflow as tf
from indicnlp.tokenize.indic_tokenize import trivial_tokenize, trivial_tokenize_indic
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np

import pandas as pd
from indicnlp.tokenize.indic_tokenize import trivial_tokenize, trivial_tokenize_indic
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import bert_score
from bert_score import score
import pickle
from itertools import chain, product
from typing import Callable, Iterable, List, Tuple
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
import pyiwn  #Hindi-wordnet
from nltk.metrics import edit_distance

# Writing Code for METEOR metric

def _generate_enums(
    hypothesis: Iterable[str],
    reference: Iterable[str],
    preprocess: Callable[[str], str] = str.lower,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Takes in pre-tokenized inputs for hypothesis and reference and returns
    enumerated word lists for each of them

    :param hypothesis: pre-tokenized hypothesis
    :param reference: pre-tokenized reference
    :preprocess: preprocessing method (default str.lower)
    :return: enumerated words list
    """
    if isinstance(hypothesis, str):
        raise TypeError(
            f'"hypothesis" expects pre-tokenized hypothesis (Iterable[str]): {hypothesis}'
        )

    if isinstance(reference, str):
        raise TypeError(
            f'"reference" expects pre-tokenized reference (Iterable[str]): {reference}'
        )

    enum_hypothesis_list = list(enumerate(map(preprocess, hypothesis)))
    enum_reference_list = list(enumerate(map(preprocess, reference)))
    return enum_hypothesis_list, enum_reference_list


def exact_match(
    hypothesis: Iterable[str], reference: Iterable[str]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    matches exact words in hypothesis and reference
    and returns a word mapping based on the enumerated
    word id between hypothesis and reference

    :param hypothesis: pre-tokenized hypothesis
    :param reference: pre-tokenized reference
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _match_enums(enum_hypothesis_list, enum_reference_list)


def _match_enums(
    enum_hypothesis_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    matches exact words in hypothesis and reference and returns
    a word mapping between enum_hypothesis_list and enum_reference_list
    based on the enumerated word id.

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        for j in range(len(enum_reference_list))[::-1]:
            if enum_hypothesis_list[i][1] == enum_reference_list[j][1]:
                word_match.append(
                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])
                )
                enum_hypothesis_list.pop(i)
                enum_reference_list.pop(j)
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def _enum_stem_match(
    enum_hypothesis_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
    stemmer: StemmerI = PorterStemmer(),
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between enum_hypothesis_list and
    enum_reference_list based on the enumerated word id. The function also
    returns a enumerated list of unmatched words for hypothesis and reference.

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    """
    stemmed_enum_hypothesis_list = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_hypothesis_list
    ]

    stemmed_enum_reference_list = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_reference_list
    ]

    return _match_enums(stemmed_enum_hypothesis_list, stemmed_enum_reference_list)


def stem_match(
    hypothesis: Iterable[str],
    reference: Iterable[str],
    stemmer: StemmerI = PorterStemmer(),
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between hypothesis and reference

    :param hypothesis: pre-tokenized hypothesis
    :param reference: pre-tokenized reference
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_stem_match(enum_hypothesis_list, enum_reference_list, stemmer=stemmer)


def _enum_wordnetsyn_match(
    enum_hypothesis_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Matches each word in reference to a word in hypothesis
    if any synonym of a hypothesis word is the exact match
    to the reference word.

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        hypothesis_syns = set(
            chain.from_iterable(
                (
                    lemma.name()
                    for lemma in synset.lemmas()
                    if lemma.name().find("_") < 0
                )
                for synset in wordnet.synsets(enum_hypothesis_list[i][1])
            )
        ).union({enum_hypothesis_list[i][1]})
        for j in range(len(enum_reference_list))[::-1]:
            if enum_reference_list[j][1] in hypothesis_syns:
                word_match.append(
                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])
                )
                enum_hypothesis_list.pop(i)
                enum_reference_list.pop(j)
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def wordnetsyn_match(
    hypothesis: Iterable[str],
    reference: Iterable[str],
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Matches each word in reference to a word in hypothesis if any synonym
    of a hypothesis word is the exact match to the reference word.

    :param hypothesis: pre-tokenized hypothesis
    :param reference: pre-tokenized reference
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :return: list of mapped tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_wordnetsyn_match(
        enum_hypothesis_list, enum_reference_list, wordnet=wordnet
    )


def _enum_align_words(
    enum_hypothesis_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    in case there are multiple matches the match which has the least number
    of crossing is chosen. Takes enumerated list as input instead of
    string input

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :return: sorted list of matched tuples, unmatched hypothesis list,
             unmatched reference list
    """
    exact_matches, enum_hypothesis_list, enum_reference_list = _match_enums(
        enum_hypothesis_list, enum_reference_list
    )

    stem_matches, enum_hypothesis_list, enum_reference_list = _enum_stem_match(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer
    )

    wns_matches, enum_hypothesis_list, enum_reference_list = _enum_wordnetsyn_match(
        enum_hypothesis_list, enum_reference_list, wordnet=wordnet
    )

    return (
        sorted(
            exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]
        ),
        enum_hypothesis_list,
        enum_reference_list,
    )


def align_words(
    hypothesis: Iterable[str],
    reference: Iterable[str],
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    In case there are multiple matches the match which has the least number
    of crossing is chosen.

    :param hypothesis: pre-tokenized hypothesis
    :param reference: pre-tokenized reference
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :return: sorted list of matched tuples, unmatched hypothesis list, unmatched reference list
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_align_words(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer, wordnet=wordnet
    )


def _count_chunks(matches: List[Tuple[int, int]]) -> int:
    """
    Counts the fewest possible number of chunks such that matched unigrams
    of each chunk are adjacent to each other. This is used to calculate the
    fragmentation part of the metric.

    :param matches: list containing a mapping of matched words (output of align_words)
    :return: Number of chunks a sentence is divided into post alignment
    """
    i = 0
    chunks = 1
    while i < len(matches) - 1:
        if (matches[i + 1][0] == matches[i][0] + 1) and (
            matches[i + 1][1] == matches[i][1] + 1
        ):
            i += 1
            continue
        i += 1
        chunks += 1
    return chunks


def single_meteor_score(
    reference: Iterable[str],
    hypothesis: Iterable[str],
    preprocess: Callable[[str], str] = str.lower,
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    """
    Calculates METEOR score for single hypothesis and reference as per
    "Meteor: An Automatic Metric for MT Evaluation with HighLevels of
    Correlation with Human Judgments" by Alon Lavie and Abhaya Agarwal,
    in Proceedings of ACL.
    https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']

    >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']


    >>> round(single_meteor_score(reference1, hypothesis1),4)
    0.6944

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(single_meteor_score(['this', 'is', 'a', 'cat'], ['non', 'matching', 'hypothesis']),4)
    0.0

    :param reference: pre-tokenized reference
    :param hypothesis: pre-tokenized hypothesis
    :param preprocess: preprocessing function (default str.lower)
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :param alpha: parameter for controlling relative weights of precision and recall.
    :param beta: parameter for controlling shape of penalty as a
                 function of as a function of fragmentation.
    :param gamma: relative weight assigned to fragmentation penalty.
    :return: The sentence-level METEOR score.
    """
    enum_hypothesis, enum_reference = _generate_enums(
        hypothesis, reference, preprocess=preprocess
    )
    translation_length = len(enum_hypothesis)
    reference_length = len(enum_reference)
    matches, _, _ = _enum_align_words(
        enum_hypothesis, enum_reference, stemmer=stemmer, wordnet=wordnet
    )
    matches_count = len(matches)
    try:
        precision = float(matches_count) / translation_length
        recall = float(matches_count) / reference_length
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        chunk_count = float(_count_chunks(matches))
        frag_frac = chunk_count / matches_count
    except ZeroDivisionError:
        return 0.0
    penalty = gamma * frag_frac**beta
    return (1 - penalty) * fmean


def meteor_score(
    references: Iterable[Iterable[str]],
    hypothesis: Iterable[str],
    preprocess: Callable[[str], str] = str.lower,
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    """
    Calculates METEOR score for hypothesis with multiple references as
    described in "Meteor: An Automatic Metric for MT Evaluation with
    HighLevels of Correlation with Human Judgments" by Alon Lavie and
    Abhaya Agarwal, in Proceedings of ACL.
    https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    In case of multiple references the best score is chosen. This method
    iterates over single_meteor_score and picks the best pair among all
    the references for a given hypothesis

    >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops', 'forever', 'hearing', 'the', 'activity', 'guidebook', 'that', 'party', 'direct']

    >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
    >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always', 'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']

    >>> round(meteor_score([reference1, reference2, reference3], hypothesis1),4)
    0.6944

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(meteor_score([['this', 'is', 'a', 'cat']], ['non', 'matching', 'hypothesis']),4)
    0.0

    :param references: pre-tokenized reference sentences
    :param hypothesis: a pre-tokenized hypothesis sentence
    :param preprocess: preprocessing function (default str.lower)
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :param alpha: parameter for controlling relative weights of precision and recall.
    :param beta: parameter for controlling shape of penalty as a function
                 of as a function of fragmentation.
    :param gamma: relative weight assigned to fragmentation penalty.
    :return: The sentence-level METEOR score.
    """
    return max(
        single_meteor_score(
            reference,
            hypothesis,
            preprocess=preprocess,
            stemmer=stemmer,
            wordnet=wordnet,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        for reference in references
    )


# Function for TER
def get_ter(ref, hyp):
    """
    This function calculates the Translation Error Rate (TER) given the reference (ref) and hypothesis (hyp) translations in Hindi.
    """
    # Tokenize the reference and hypothesis translations using the IndicTokenizer
    # tokenizer = IndicTokenizer('hi')
    ref_tokens = trivial_tokenize(ref)
    hyp_tokens = trivial_tokenize(hyp)

    # Compute the edit distance between the tokenized translations using the edit_distance function
    edit_dis = edit_distance(ref_tokens, hyp_tokens)

    # Compute the TER score as the edit distance divided by the length of the reference translation
    ter_score = edit_dis / len(ref_tokens)

    return ter_score

# TER scores can be greater than 1
def normalize_ter_scores(scores):
  res = [1-min(1, score) for score in scores]
  return res


# BLEU scores function 
def get_bleu(cand, ref):
    c_tokens = trivial_tokenize(cand.lower(),lang='hi')
    r_tokens = trivial_tokenize(ref.lower(),lang='hi')

    # considering 3-gram overlap
    weights=(1./3., 1./3., 1./3.)
    bl_sc = sentence_bleu([r_tokens], c_tokens, weights=weights)
    return bl_sc

# Get Function for Meteor Score
def get_meteor(ref, cand):
    references = [ref.split(" ")]
    predictions = cand.split(" ")
    m_score = meteor_score(references = references, hypothesis=predictions, wordnet = pyiwn.IndoWordNet())
    return m_score

# function to get BERT Score
# list of candidate and reference
def get_bert(cand, ref):
    bert_score = score(cand, ref, lang='hi')
    return bert_score


# write list to binary file
def write_list(a_list, name):
    # store list in binary file so 'wb' mode
    with open(name, 'wb') as fp:
        pickle.dump(a_list, fp)
        # print('Done writing list into a binary file')

# Read list to memory
def read_list(name):
    # for reading also binary mode is important
    with open(name, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

# Load the ai4bharat/indic-bert model and tokenizer
model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(sentence):
  # Tokenize a Hindi sentence
  # sentence = "मैं भारत से प्यार करता हूँ"
  tokens = trivial_tokenize(sentence, lang='hi')

  # Convert tokens to token IDs and add special tokens
  tokens_with_special = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokens))

  # Convert token IDs to a tensor
  tokens_tensor = torch.tensor([tokens_with_special])

  # Generate BERT embeddings for the tokens
  with torch.no_grad():
      outputs = model(tokens_tensor)
      embeddings = outputs.last_hidden_state  # shape: (1, seq_len, hidden_size)

  # Average the token-level embeddings across the entire sentence
  sentence_embedding = torch.mean(embeddings, dim=1)  # shape: (1, hidden_size)

  # Convert the tensor to a numpy array
  sentence_embedding = sentence_embedding.squeeze().numpy()

  return sentence_embedding


def get_cnn_rank(ref, cand):
    x = []
    temp = []
    ref_embeddings = get_embedding(ref)
    cand_embeddings = get_embedding(cand)
    temp.extend(ref_embeddings)
    temp.extend(cand_embeddings)
    x.append(temp)
    # reload the pickle file
    bleu = get_bleu(cand, ref)
    ter = normalize_ter_scores([get_ter(cand, ref)])[0]
    m_score = get_meteor(cand,ref)
    P, R, F = get_bert([cand],[ref])
    
    pca = pickle.load(open("pca_704.pkl",'rb'))
    X = pca.transform(x)
    x_temp = np.zeros((1,704))
    x_temp[:, 0] = F
    x_temp[:, 1] = bleu
    x_temp[:, 2] = ter
    x_temp[:, 3] = m_score
    x_temp[:,4:] = X
    model = tf.keras.models.load_model('CNN_WMT')
    y = model.predict(x_temp)
    prediction = y[0]
    rank = 0
    for i in range(5):
        mx = max(prediction)
        if prediction[i]==mx:
            rank = i+1
            break
    print("rank is : ", rank)
    return rank