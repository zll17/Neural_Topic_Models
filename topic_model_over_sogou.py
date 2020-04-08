def show_topics(model=lda_model,topn=20,n_topic=10,fix_topic=None,showWght=False):
    global vocab
    topics = []
    def show_one_tp(tp_idx):
        if showWght:
            return [(vocab.id2token[t[0]],t[1]) for t in lda_model.get_topic_terms(tp_idx)]
        else:
            return [vocab.id2token[t[0]] for t in lda_model.get_topic_terms(tp_idx)]
    if fix_topic is None:
        for i in range(n_topic):
            topics.append(show_one_tp(i))
    else:
        topics.append(show_one_tp(fix_topic))
    return topics

