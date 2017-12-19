from dataset_transformer import prepare_class

prepare_class("../aclImdb/test/neg/", "../aclImdb/imdb.vocab", "../aclImdb/test/neg/prepared_test_neg")
prepare_class("../aclImdb/test/pos/", "../aclImdb/imdb.vocab", "../aclImdb/test/neg/prepared_test_pos")
prepare_class("../aclImdb/train/neg/", "../aclImdb/imdb.vocab", "../aclImdb/test/neg/prepared_neg")
prepare_class("../aclImdb/train/pos/", "../aclImdb/imdb.vocab", "../aclImdb/test/neg/prepared_pos")

