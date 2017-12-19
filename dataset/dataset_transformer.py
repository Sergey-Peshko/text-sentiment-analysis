import glob
import os
import re


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def prepare_class(class_path, vocab_path, name):
    path = os.path.join(class_path, '*txt')

    files = glob.glob(path)

    regex1 = re.compile(r"(<\\w+ \\/>)", re.IGNORECASE)
    regex2 = re.compile(r"[\\W\\d\\s&&[^']]", re.IGNORECASE)
    regex3 = re.compile(r" +", re.IGNORECASE)

    counter = 0

    with open(name, "w") as output:

        with open(vocab_path) as f:
            vocab = f.read().splitlines()

        for file in files:

            print(class_path + counter.__str__())
            counter += 1

            points = int(find_between(file, "_", ".txt"))

            if points > 5:
                output.write("1")
            else:
                output.write("-1")

            with open(file) as f:
                text = f.read().splitlines()[0]

            replaced = regex1.sub(" ", text)
            replaced = regex2.sub(" ", replaced)
            replaced = regex3.sub(" ", replaced)
            replaced = replaced.lower()

            words = replaced.split(" ")

            image = {}

            for word in words:
                try:
                    index = vocab.index(word)
                except ValueError:
                    index = -1
                if index is not -1:
                    value = image.get(index)
                    if value is None:
                        image[index] = 0
                    else:
                        image[index] += 1

            for key, val in image.items():
                output.write(" {0}:{1}".format(key, val))

            output.write("\n")
