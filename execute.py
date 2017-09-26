from extracting import extracting
from learning import learning

default_comment_directory = "./comment_source/"
filtered_comment_directory = "./filtered_comment_source_other/"


def execute(extracting_dic, learning_dic):

    extracting(**extracting_dic)
    learning(**learning_dic)


def main():
    count = 0

    extracting_dic = {
        "word_class": ["形容詞"],
        "max_features": 30,
        "vector": 'count',
        "comment_directory_path": default_comment_directory,
        "output_feature_name": "adj_30_count_standard.csv"
    }

    learning_dic = {
        "output_feature_name": "adj_30_count_standard.csv",
        "result_name": "feature_values.csv",
        "is_standard": True
    }

    word_class_array = [["形容詞"], ["形容詞", "名詞"]]
    max_features_array = [30, 100]
    vector_array = ["tfidf", "count"]
    comment_directory_path_array = [
        default_comment_directory,
        filtered_comment_directory
    ]
    is_standard_array = [True, False]

    for word_class in word_class_array:
        for max_features in max_features_array:
            for vector in vector_array:
                for comment_directory_path in comment_directory_path_array:
                    for is_standard in is_standard_array:

                        first_name = ""
                        if word_class == ["形容詞"]:
                            first_name = "adj"
                        else:
                            first_name = "all"

                        filter_name = ""
                        if comment_directory_path == default_comment_directory:
                            filter_name = "nf"
                        else:
                            filter_name = "f"

                        standard_name = ""
                        if is_standard is True:
                            standard_name = "standard"
                        else:
                            standard_name = "nostandard"
                        # adj, filter, feature, vector, standard
                        feature_name = "%s_%s_%s_%s_%s_feature.csv" % (
                            first_name, filter_name, max_features, vector, standard_name)

                        result_name = "%s_%s_%s_%s_%s_result.csv" % (
                            first_name, filter_name, max_features, vector, standard_name)

                        extracting_dic = {
                            "word_class": word_class,
                            "max_features": max_features,
                            "vector": vector,
                            "comment_directory_path": comment_directory_path,
                            "output_feature_name": './results/features/' + feature_name
                        }

                        learning_dic = {
                            "output_feature_name": './results/features/' + feature_name,
                            "result_name": './results/accuracy/' + result_name,
                            "is_standard": is_standard
                        }
                        execute(extracting_dic, learning_dic)
                        count += 1
                        print(str(count) + "/32 finished!")


if __name__ == '__main__':
    main()
