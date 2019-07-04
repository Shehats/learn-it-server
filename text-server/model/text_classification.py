import pandas as pd
import os
import chakin
import numpy as np
import tensorflow as tf
import json
from string import punctuation
from collections import defaultdict


CHAKIN_INDEX = 17
NUMBER_OF_DIMENSIONS = 25
SUBFOLDER_NAME = "glove.twitter.27B"
DATA_FOLDER = "embeddings"
PRE_FIX = "glove"
prefix = SUBFOLDER_NAME + "." + str(NUMBER_OF_DIMENSIONS) + "d"
TF_EMBEDDINGS_FILE_NAME = os.path.join(DATA_FOLDER, prefix + ".ckpt")
DICT_WORD_TO_INDEX_FILE_NAME = os.path.join(DATA_FOLDER, prefix + ".json")


"""
    load_from_file: downloads data from chakin,
    then returns word indicies
"""
def load_from_file(chakin_index, nb_dims, pre_fix, sub_folder, root_folder, save_dir):
    zip_file = os.path.join(root_folder, "{}.zip".format(sub_folder))
    zip_file_alt = pre_fix + zip_file[5:]
    unzip_folder = os.path.join(root_folder, sub_folder)
    if sub_folder[-1] == "d":
        golve_fname = os.path.join(unzip_folder, "{}.txt".format(sub_folder))
    else:
        golve_fname = os.path.join(unzip_folder, "{}.{}d.txt".format(sub_folder, nb_dims))
    
    if not os.path.exists(zip_file) and not os.path.exists(unzip_folder):
        print("Downloading embeddings to '{}'".format(zip_file))
        chakin.download(number=chakin_index, save_dir=save_dir)
    else:
        print("Embeddings has already been downloaded")
    
    if not os.path.exists(unzip_folder):
        import zipfile
        if not os.path.exists(zip_file) and os.path.exists(zip_file_alt):
            zip_file = zip_file_alt
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            print("Extracting embeddings to '{}'".format(unzip_folder))
            zip_ref.extractall(unzip_folder)
    else:
        print("Embeddings has already been extracted.")
    
    # Load indicies from disk
    word_to_embedding_dict = dict()
    index_to_embedding = []
    num_rep = 0
    j = 0
    with open(golve_fname, "r") as golve_file:
        for i, line in enumerate(golve_file):
            split = line.split(" ")
            word = split[0]
            representation = split[1:]

            representation = np.array(
                [float(val) for val in representation]
            )
            word_to_embedding_dict[word] = i
            index_to_embedding.append(representation)

            if num_rep == 0:
                num_rep = len(representation)
            
            j = i
    
    _word_not_found = np.array([0*0] * num_rep)

    j += 1
    word_to_embedding_dict["UNKNOWN"] = j
    index_to_embedding = np.array(_word_not_found + [_word_not_found])
    return word_to_embedding_dict, index_to_embedding

def init_embeddings(index_to_embedding, tf_embedding_file_name):
    batch_size = None
    # Create a new graph
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    # load initinal embedding
    tf_embedding = tf.Variable(
        tf.constant(0.0, shape=index_to_embedding.shape),
        trainable=False,
        name="Embedding"
    )

    # load initial indices
    tf_word_ids = tf.placeholder(tf.int32, shape=[batch_size])

    tf_word_representation_layer = tf.nn.embedding_lookup(
        params=tf_embedding,
        ids=tf_word_ids
    )

    tf_embedding_placeholder = tf.placeholder(tf.float32, shape=index_to_embedding.shape)

    tf_embedding_init = tf_embedding.assign(tf_embedding_placeholder)

    _ = sess.run(
        tf_embedding_init,
        feed_dict={
            tf_embedding_placeholder: index_to_embedding
        }
    )

    variables_to_save = [tf_embedding]
    embedding_saver = tf.train.Saver(variables_to_save)
    embedding_saver.save(sess, save_path=tf_embedding_file_name)
    print("TF embeddings saved to '{}'.".format(tf_embedding_file_name))

    # Create a builder to export the model
    builder = tf.saved_model.builder.SavedModelBuilder("export")
    # Tag the model in order to be capable of restoring it specifying the tag set
     # clear_device=True in order to export a device agnostic graph.
    builder.add_meta_graph_and_variables(sess, ["tag"], clear_devices=True)
    builder.save()

    sess.close()

    del index_to_embedding
    return sess

def load_word_to_index(dict_word_to_index_file_name):
    word_to_index = {}
    with open(dict_word_to_index_file_name) as f:
        word_to_index = json.load(f)
    _last_index = len(word_to_index) - 2
    word_to_index["UNKNOWN"] = _last_index
    return word_to_index

def load_embedding_tf(session, word_to_index, tf_embedding_file_path, nb_dims):
    print(len(word_to_index))
    tf_embedding = tf.Variable(
        tf.constant(0.0, shape=[len(word_to_index), nb_dims]),
        trainable=False,
        name="Embedding"
    )

    # Restore the embedding from disks to TensorFlow
    variable_to_restore = [tf_embedding]
    embedding_saver = tf.train.Saver(variable_to_restore)
    embedding_saver.restore(session, save_path=tf_embedding_file_path)
    return tf_embedding

def compute_similarity(tf_word_rep_x, tf_word_rep_y):
    x_normalized = tf.nn.l2_normalize(tf_word_rep_x, axis=-1)
    y_normalized = tf.nn.l2_normalize(tf_word_rep_y, axis=-1)

    similarity = tf.reduce_sum(
        tf.multiply(x_normalized, y_normalized),
        axis=-1
    )

    return similarity

def init_model():
    # load data from disk then get embeddings
    word_to_embedding_dict, index_to_embedding = load_from_file(chakin_index=CHAKIN_INDEX, 
                                                                nb_dims=NUMBER_OF_DIMENSIONS, 
                                                                pre_fix=PRE_FIX, 
                                                                sub_folder=SUBFOLDER_NAME, 
                                                                root_folder=DATA_FOLDER,
                                                                save_dir=DATA_FOLDER)

    _sess = init_embeddings(index_to_embedding=index_to_embedding,
                            tf_embedding_file_name=TF_EMBEDDINGS_FILE_NAME)
    
    with open(DICT_WORD_TO_INDEX_FILE_NAME, "w") as f:
        json.dump(word_to_embedding_dict, f)
    
    print("word_to_index dict saved to '{}'.".format(DICT_WORD_TO_INDEX_FILE_NAME))

def create_model():
    batch_size = None
    if not os.path.exists(TF_EMBEDDINGS_FILE_NAME) or not os.path.exists(DICT_WORD_TO_INDEX_FILE_NAME):
        init_model()
    
    tf.reset_default_graph()
    # Create new session
    _sess = tf.InteractiveSession()
    _word_to_index = load_word_to_index(DICT_WORD_TO_INDEX_FILE_NAME)
    _tf_embedding = load_embedding_tf(_sess,
                                    _word_to_index, 
                                    TF_EMBEDDINGS_FILE_NAME, 
                                    NUMBER_OF_DIMENSIONS)

    # Input to the graph where word IDs can be sent in batch. Look at the "shape" args:                          
    tf_word_x_id = tf.placeholder(tf.int32, shape=[1])
    tf_word_y_ids = tf.placeholder(tf.int32, shape=[batch_size])
    
    # Conversion of words to a representation
    tf_word_rep_x = tf.nn.embedding_lookup(
        params=_tf_embedding, ids=tf_word_x_id
    )

    tf_word_rep_y = tf.nn.embedding_lookup(
        params=_tf_embedding, ids=tf_word_y_ids
    )

    text_similarities = compute_similarity(
        tf_word_rep_x=tf_word_rep_x,
        tf_word_rep_y=tf_word_rep_y
    )

    model_dict = {
        "sess": _sess,
        "text_similarities": text_similarities,
        "word_to_index": _word_to_index,
        "tf_word_x_id": tf_word_x_id,
        "tf_word_y_ids": tf_word_y_ids
    }

    return model_dict

def sentence_to_word_ids(sentence, word_to_index):
    # Separating punctuation from words:
    for punctuation_character in punctuation:
        sentence = sentence.replace(punctuation_character, " {} ".format(punctuation_character))
    
    # Removing double spaces and lowercasing:
    sentence = sentence.replace("  ", " ").replace("  ", " ").lower().strip()
    # Splitting on every space:
    split_sentence = sentence.split(" ")
    # Converting to IDs:
    ids = [word_to_index[w.strip()] for w in split_sentence]
    return ids, split_sentence

def predict(model_dict, word_x, word_y):
    word_x_id, _ = sentence_to_word_ids(word_x, model_dict["word_to_index"])
    word_y_ids, split_sentence = sentence_to_word_ids(word_y, model_dict["word_to_index"])

    prediction = model_dict["sess"].run(
        model_dict["text_similarities"],
        feed_dict={
            model_dict["tf_word_x_id"]: word_x_id,
            model_dict["tf_word_y_ids"]: word_y_ids
        }
    )
    return prediction, split_sentence
