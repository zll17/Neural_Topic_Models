from gensim.models.coherencemodel import CoherenceModel

def get_topic_words(model,topn=15,n_topic=10,vocab=None,fix_topic=None,showWght=False):
    topics = []
    def show_one_tp(tp_idx):
        if showWght:
            return [(vocab.id2token[t[0]],t[1]) for t in model.get_topic_terms(tp_idx,topn=topn)]
        else:
            return [vocab.id2token[t[0]] for t in model.get_topic_terms(tp_idx,topn=topn)]
    if fix_topic is None:
        for i in range(n_topic):
            topics.append(show_one_tp(i))
    else:
        topics.append(show_one_tp(fix_topic))
    return topics

def calc_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words,[]))
    n_total = len(topic_words) * len(topic_words[0])
    topic_div = len(vocab) / n_total
    return topic_div

def calc_topic_coherence(topic_words,docs,dictionary,emb_path=None):
    # Computing the C_V score
    cv_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_v')
    #cv_coherence_score = cv_coherence_model.get_coherence_per_topic()
    cv_score = cv_coherence_model.get_coherence()
    
    # Computing the C_W2V score
    try:
        if os.path.exists(emb_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(emb_path)
            w2v_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_w2v',keyed_vectors=keyed_vectors)
        else:
            w2v_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_w2v')
        #w2v_coherence_score = w2v_coherence_model.get_coherence_per_topic()
        w2v_score = w2v_coherence_model.get_coherence()
    except:
        #In case of OOV Error
        w2v_score = None
    
    # Computing the C_UCI score
    c_uci_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_uci')
    #c_uci_coherence_score = c_uci_coherence_model.get_coherence_per_topic()
    c_uci_score = c_uci_coherence_model.get_coherence()
    
    # Computing the C_NPMI score
    c_npmi_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_npmi')
    #c_npmi_coherence_score = c_npmi_coherence_model.get_coherence_per_topic()
    c_npmi_score = c_npmi_coherence_model.get_coherence()

    return cv_score, w2v_score, c_uci_score, c_npmi_score
