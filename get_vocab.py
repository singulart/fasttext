#!/usr/bin/env python

from fasttext import load_model
import argparse
import errno

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Print words or labels and frequency of a model's dictionary"
        )
    )
    parser.add_argument(
        "model",
        help="Model to use",
    )
    parser.add_argument(
        "-l",
        "--labels",
        help="Print labels instead of words",
        action='store_true',
        default=False,
    )
    args = parser.parse_args()

    f = load_model(args.model)
    if args.labels:
        words, freq = f.get_labels(include_freq=True)
    else:
        words, freq = f.get_words(include_freq=True)
    for w, f in zip(words, freq):
        try:
            print(w + "\t" + str(f))
        except IOError as e:
            if e.errno == errno.EPIPE:
                pass
