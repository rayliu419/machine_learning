import os


def prepare_translation_raw_data_for_task():
    """
    返回diabetes的raw data
    :return: pandas data frame
    """
    print("load data")
    data_file = os.path.dirname(__file__) + "/input_data/eng-fra.txt"
    line_index = 0
    X_raw = []
    Y_raw = []
    with open(data_file) as infile:
        for line in infile:
            line_index += 1
            line = line.strip()
            eng, fra = line.split("\t")
            X_raw.append(eng)
            Y_raw.append(fra)

if __name__ == "__main__":
    prepare_translation_raw_data_for_task()
