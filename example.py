'''
This is a minimal example using most of the default parameters.
See xx.py for a full example.
'''
from data_utils import *
from dataset import DocDataset
from models import ETM

# Create a corpus from a file with lines of texts.
text = file_tokenize("data/zhdd_lines.txt", "zh")
dictionary = build_dictionary(text)
bows, docs = convert_to_BOW(text, dictionary)
corpus = DocDataset(dictionary, bows, docs)

# # or:
# corpus = DocDataset()
# corpus.create("data/zhdd_lines.txt", "zh", use_tfidf=False)

# Save corpus
# Need to save corpus, otherwise cannot train from checkpoint
corpus.save("data/zhdd_dev")

# # Load corpus
# corpus = DocDataset()
# corpus.load("data/zhdd_dev")
# dictionary = corpus.dictionary  # for debug

# Train the model on the corpus
model = ETM(bow_dim=corpus.vocabsize, n_topic=10)
model.train(train_data=corpus, test_data=corpus, num_epochs=3, log_every=1)  # num_epochs=10, log_every=1 for debug
# TODO: remove test_data parameter.
# TODO: remove train with initilization - [to discuss]

# Evaluate the model on the training corpus
model.evaluate(test_data=corpus)

# Save a model to disk, or reload a pre-trained model
model.save("model.ckpt")
model.load("model.ckpt")

# Query, the model using new, unseen documents
other_texts = [
    ['晚饭', '喝点', '啤酒', '诱人'],
    ['俯卧撑', '小菜一碟', '信不信', '分钟'],
    ['溜冰鞋', '新的', '拿到', '社区', '联盟'],
    [],
    None
]
other_bows, other_docs = convert_to_BOW(other_texts, dictionary, keep_empty_doc=True)
unseen_doc = other_bows[0]

doc_topics = model[unseen_doc]
print(doc_topics)

# Get the topic distribution for the given document.
doc_topics = model.get_document_topics(unseen_doc, minimum_probability=0.0004)
print(doc_topics)

# Inference for a dataset
docs, embeds = model.inference_dataset(corpus)
print(docs, embeds)

#### show topics of a model
# Print the most significant topics
model.print_topics(num_topics=5, num_words=10)

# Print a topic (topic as words)
topicno=0
model.print_topic(topicno, topn=10)

# Print a topic (topic as id)
topicid=0
print(model.get_topic_terms(topicid, topn=10))


'''
# Update the model by incrementally training on the new corpus
# TO DISCUSS
model.update()

# Calculate the difference in topic distributions between two models
m1.diff(m2)

# Get the most relevant topics to the given word.
model.get_term_topics(word_id, minimum_probability=None)

# Get the term-topic matrix learned during inference.
get_topics()

# Get the topics with the highest coherence score the coherence for each topic.
top_topics(corpus=None, texts=None, dictionary=None, window_size=None, coherence='u_mass', topn=20, processes=-1)
'''