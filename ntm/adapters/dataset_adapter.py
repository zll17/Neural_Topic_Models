from dataset import DocDataset, TestData


def _effective_rebuild(data_cfg):
    # Align with CLI: --no_rebuild => rebuild=False; otherwise use rebuild flag
    if getattr(data_cfg, "no_rebuild", False):
        return False
    return data_cfg.rebuild


def build_train_dataset(data_cfg):
    rebuild = _effective_rebuild(data_cfg)
    dataset = DocDataset(
        taskname=data_cfg.taskname,
        lang=data_cfg.lang,
        no_below=data_cfg.no_below,
        no_above=data_cfg.no_above,
        rebuild=rebuild,
        use_tfidf=data_cfg.use_tfidf,
    )
    if data_cfg.auto_adj:
        no_above = dataset.topk_dfs(topk=20)
        dataset = DocDataset(
            taskname=data_cfg.taskname,
            lang=data_cfg.lang,
            no_below=data_cfg.no_below,
            no_above=no_above,
            rebuild=rebuild,
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
