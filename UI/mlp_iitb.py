import pickle
import tensorflow as tf
from indicnlp.tokenize.indic_tokenize import trivial_tokenize, trivial_tokenize_indic
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np

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


def get_mlp_score(ref, cand):
    x = []
    temp = []
    ref_embeddings = get_embedding(ref)
    cand_embeddings = get_embedding(cand)
    temp.extend(ref_embeddings)
    temp.extend(cand_embeddings)
    x.append(temp)
    # reload the pickle file
    pca = pickle.load(open("pca.pkl",'rb'))
    X = pca.transform(x)
    model = tf.keras.models.load_model('MLP_ep_300_IITB_data')
    y = model.predict(X)
    print(y)
    return np.mean(y[0])