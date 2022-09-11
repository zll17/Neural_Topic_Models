'''
This is a minimal example using most of the default parameters.
See xx.py for a full example.
TODO: Add save_dir to save tempf files.
'''
from data_utils import *
from dataset import DocDataset
from models import ETM

# Create a corpus from a file with lines of texts.
text = file_tokenize("data/zhdd_lines.txt", "zh")
dictionary = build_dictionary(text)
bows, docs = convert_to_BOW(text, dictionary)
corpus = DocDataset(dictionary, bows, docs)

# Train the model on the corpus
model = ETM(bow_dim=corpus.vocabsize, n_topic=10, taskname="zhdd")
model.train(train_data=corpus, test_data=corpus)  # to clarify: train saves ckpt and evaluates every log_every steps
'''
# Evaluate the model
model.evaluate(test_data=corpus)

# Save a model to disk, or reload a pre-trained model
tmp_file = "model.ckpt"
model.save_model(tmp_file)  # To add
model.load_model(tmp_file)

# Query, the model using new, unseen documents
other_corpus = TestDocDataset(dictionary=dictionary, txtPath="example_lines.txt", lang="zh")
unseen_doc = other_corpus[0]
vector = model.inference(doc_tokenized=unseen_doc, dictionary=dictionary)

# Update the model by incrementally training on the new corpus
# TO DISCUSS
model.update()

# Calculate the difference in topic distributions between two models
m1.diff(m2)

# Get the topic distribution for the given document.
model.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)

# Get the most relevant topics to the given word.
model.get_term_topics(word_id, minimum_probability=None)

# Get the representation for a single topic. 
# Words the integer IDs, in constrast to show_topic() that represents words by the actual strings.
model.get_topic_terms(topicid, topn=10)

# Get the term-topic matrix learned during inference.
get_topics()

# Get a single topic as a formatted string.
print_topic(topicno, topn=10)

# Get the most significant topics (alias for show_topics() method).
print_topics(num_topics=20, num_words=10)

# Get the representation for a single topic. 
# Words here are the actual strings, in constrast to get_topic_terms() that represents words by their vocabulary ID.
show_topic(topicid, topn=10)

# Get a representation for selected topics.
show_topics(num_topics=10, num_words=10, log=False, formatted=True)

# Get the topics with the highest coherence score the coherence for each topic.
top_topics(corpus=None, texts=None, dictionary=None, window_size=None, coherence='u_mass', topn=20, processes=-1)
'''