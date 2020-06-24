import os


def prepare_chinese_poi_raw_data_for_task(seg=False, linenum=1000):
    """
    返回chinese的raw data
    如果seg=False, 则整句返回
    如果seg=True, 则分字返回，对于中文一个个字。
    :return: pandas data frame
    """
    print("load data")
    data_file = os.path.dirname(__file__) + "/input_data/test_10000"
    line_index = 0
    data = []
    with open(data_file) as infile:
        for line in infile:
            line_index += 1
            if (line_index == linenum):
                break
            line = line.strip()
            if not seg:
                data.append(line)
            else:
                cur = []
                cur.extend(line)
                data.append(cur)
    return data


if __name__ == "__main__":
    pois_seg = prepare_chinese_poi_raw_data_for_task(True)
    for poi in pois_seg[0:50]:
        print(poi)
    pois_no_seg = prepare_chinese_poi_raw_data_for_task(False)
    for poi in pois_no_seg[0:50]:
        print(poi)

