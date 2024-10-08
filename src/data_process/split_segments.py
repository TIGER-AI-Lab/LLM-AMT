import os
import json
from tqdm import tqdm


def load_books():
    data = {}
    seg_id = 0
    books_dir = r"data/textbooks_en/"
    for file in tqdm(os.listdir(books_dir)):
        if file.endswith(".txt") and not file.startswith("."):
            with open(os.path.join(books_dir, file), "r", encoding="utf-8") as fi:
                for line in tqdm(fi.readlines()):
                    segs = split_lines(line)
                    for seg in segs:
                        if seg not in data:
                            data[seg] = {"seg_id": seg_id, "book_name": file}
                            seg_id += 1
    return data


def split_lines(line):
    for i in range(10):
        curr = " "*(11 - i)
        line = line.replace(curr, "\t")
    res = line.split("\t")
    res = post_process(res)
    return res


def post_process(segs):
    res = []
    for seg in segs:
        if len(seg) <= 10:
            continue
        words = seg.split(" ")
        if len(words) <= 3:
            continue
        if len(words) <= 1000:
            res.append(seg)
        else:
            sentences = seg.split(". ")
            i = 0
            curr = []
            while i < len(sentences):
                sentence = sentences[i]
                if len(sentence.split(" ")) > 1000:
                    sentence = " ".join(sentence.split(" ")[:1000])
                if len(". ".join(curr).split(" ")) + len(sentence.split(" ")) < 1000:
                    curr.append(sentence)
                else:
                    res.append(". ".join(curr))
                    curr = [sentence]
                i += 1
    return res


def save_data(data):
    output_file = r"data/segmented_textbooks/segments_of_books_0704.json"
    with open(output_file, "w", encoding="utf-8") as fo:
        fo.write(json.dumps(data, ensure_ascii=False))
    return


def main():
    data = load_books()
    # data = process(data)
    save_data(data)


if __name__ == '__main__':
    main()

