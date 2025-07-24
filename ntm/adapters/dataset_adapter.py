from dataset import DocDataset, TestData


def build_train_dataset(data_cfg):
    dataset = DocDataset(
        taskname=data_cfg.taskname,
        lang=data_cfg.lang,
        no_below=data_cfg.no_below,
        no_above=data_cfg.no_above,
        rebuild=data_cfg.rebuild,
        use_tfidf=data_cfg.use_tfidf,
    )
    if data_cfg.auto_adj:
        no_above = dataset.topk_dfs(topk=20)
        dataset = DocDataset(
            taskname=data_cfg.taskname,
            lang=data_cfg.lang,
            no_below=data_cfg.no_below,
            no_above=no_above,
            rebuild=data_cfg.rebuild,
            use_tfidf=data_cfg.use_tfidf,
        )
    return dataset


def build_test_dataset(dictionary, txt_path, data_cfg):
    return TestData(
        dictionary=dictionary,
        txtPath=txt_path,
        lang=data_cfg.lang,
        no_below=data_cfg.no_below,
        no_above=data_cfg.no_above,
        use_tfidf=data_cfg.use_tfidf,
    )
