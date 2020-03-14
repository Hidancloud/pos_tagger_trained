import torch
import sys
from tagger_train import prepare_sequence, get_tag, all_chars, LSTMTagger


def append_word(sentence, maxlen):
# making all words of the same length for convolution layer
    new_sentence = list()
    for word in sentence:
        new_word = word
        for j in range(maxlen - len(word)):
            new_word += '#'
        new_sentence.append(new_word)
    return new_sentence


def tag_sentence(test_file, model_file, out_file):
    word_to_ix, ix_to_tag, maxlen, model = torch.load(model_file)
    test_data = open(test_file, "r")
    outF = open(out_file, "w")

    for line in test_data:
        before_append_sentence = line.split()
        sentence = append_word(before_append_sentence, int(maxlen))
        test_input = prepare_sequence(sentence, word_to_ix)
	# input_char is vector X_ij for chars of this sentence
        input_char = torch.tensor([[all_chars.index(c) for c in word] for word in sentence])
        tag_scores = model(test_input, input_char)
        output = get_tag(sentence, tag_scores, ix_to_tag)
	# removing odd '#' for the output
        output = output.replace('#', '')
        outF.write(output)
        outF.write("\n")
    outF.close()


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
