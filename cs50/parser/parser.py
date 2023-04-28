import nltk
import sys
import os

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP
S -> NP VP PP
S -> S Conj S
S -> S Conj VP
NP -> N | Det NP | Adj NP | Det Adj NP | NP Adv | NP PP
VP -> V | VP NP | Adv VP | V PP | V Adv
PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    sentences = []
    if len(sys.argv) == 2:
        if os.path.isdir(sys.argv[1]):
            for filename in sorted(os.listdir(sys.argv[1]), key=lambda x: int(x.split(".")[0])):
                with open(os.path.join(sys.argv[1], filename)) as f:
                    sentences.append(f.read())
        else:
            with open(sys.argv[1]) as f:
                s = f.read()
                sentences.append(s)

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")
        sentences.append(s)

    for s in sentences:
        # Convert input into list of words
        s = preprocess(s)

        # Attempt to parse sentence
        try:
            trees = list(parser.parse(s))
        except ValueError as e:
            print(e)
            return
        if not trees:
            print("Could not parse sentence.")
            return

        # Print each tree with noun phrase chunks
        for tree in trees:
            tree.pretty_print()

            print("Noun Phrase Chunks")
            for np in np_chunk(tree):
                print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = nltk.word_tokenize(sentence.lower())
    words = [word for word in words if any(char.isalpha() for char in word)]
    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    noun_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            if not any(subtree.subtrees(lambda t: t.label() == "NP")):
                noun_phrases.append(subtree)
    return noun_phrases


if __name__ == "__main__":
    main()
