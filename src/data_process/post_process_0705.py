import json

id2doc = {}
with open(r"../knowledge_segmentation/data/segments_of_books_0704.json", "r", encoding="utf-8") as fii:
    docs = json.load(fii)
for k, v in docs.items():
    id2doc[v["seg_id"]] = k


def post_process():
    retrieve_result_list = ["data/transfer_data/sparse_result_0705.json",
                            "data/transfer_data/dense_result_0705.json",
                            "data/transfer_data/sparse_rerank_result_0705.json",
                            "data/transfer_data/dense_rerank_result_0705.json",
                            "../late_interaction/data/colbert_result_0706.json"]
    data = []
    with open(r"data/dev.json", "r", encoding="utf-8") as fi:
        for line in fi.readlines():
            data.append(json.loads(line))
    for result_file in retrieve_result_list:
        data = single_process(data, result_file)
    with open(r"data/dev_with_retrieve_0706.json", "w", encoding="utf-8") as fo:
        for each in data:
            fo.write(json.dumps(each, ensure_ascii=False))
            fo.write("\n")


def single_process(data, result_file):
    re_key = result_file.split("/")[-1].replace(".json", "")
    with open(result_file, "r", encoding="utf-8") as fi:
        temp = json.load(fi)
    for i, each in enumerate(data):
        q_id = "q" + str(i)
        re_result = temp.get(q_id, {})
        doc_list = list(re_result.keys())
        doc_list = sorted(doc_list, key=lambda x: -re_result[x])
        data[i][re_key] = []
        for doc in doc_list:
            content = id2doc[get_id(doc)]
            score = re_result[doc]
            data[i][re_key].append({"content": content, "score": score, "doc_id": doc})
    return data


def get_id(id_text):
    return int(id_text.replace("q", "").replace("doc", ""))


def main():
    post_process()


if __name__ == "__main__":
    main()


