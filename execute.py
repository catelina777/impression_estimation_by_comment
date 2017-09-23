from extracting import extracting
from learning import learning


def execute(extracting_dic, learning_dic):
    extracting(**extracting_dic)
    learning(**learning_dic)


def main():
    extracting_dic = {
        "word_class": ["形容詞"],
        "max_features": 30,
        "vector": 'count',
        "comment_directory_path": "./comment_source/",
        "output_feature_name": "feature_values.csv"
    }

    learning_dic = {
        "output_feature_name": "feature_values.csv",
        "result_name": "feature_values.csv",
        "is_standard": True
    }

    execute(extracting_dic, learning_dic)


if __name__ == '__main__':
    main()
