
#TODO (jbg): There was also a vocab file defined in the toytokenizer class, let's see if that can be moved over.  We first need to see who is using this vocab.

from typing import Iterable, Union, Tuple

class Vocab:
    def __init__(self, counts, specials=[], max_tokens=-1, vocab_cutoff=0):
        self.default_index = None
        self.final = False
        self.lookup = {}
        self.reverse_lookup = {}
        
        for idx, word in enumerate(specials):
            self.lookup[word] = idx
            self.reverse_lookup[idx] = word

        if max_tokens <= 0:
            max_tokens = len(counts)

        lookup = max(self.lookup.values())

        # Minus 1 because of the UNK token
        valid_tokens = list(counts.most_common(max_tokens - 1))
        
        # Sort so that it's the most frequent tokens, breaking tie by alphebetical order
        valid_tokens.sort(key=lambda x: (-x[1], x[0]))
        for word, count in valid_tokens:
            if vocab_cutoff <= 0 or (vocab_cutoff > 0 and count > vocab_cutoff):
                lookup += 1
                self.lookup[word] = lookup
                self.reverse_lookup[lookup] = word

    def add(self, word: str, idx: int=-1) -> int:
        """
        Add a word to the vocab and return its index.

        Args:
            word: The word to add to the vocabulary
            idx: The index it should have.  WARNING: If this is already in the vocabulary, it will be overwritten.
        Returns:
            The index of the word after it was added (useful if you don't specify the index).
        """
        assert not self.final, "Vocabulary already finalized, cannot add more words"
        
        if idx == -1:
            idx = max(self._id_to_word.keys()) + 1

        self.lookup[idx] = word
        self.reverse_lookup[word] = idx

        return idx

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """

        # Add code to generate the vocabulary that the vocab lookup
        # function can use!
        assert self.default_index is not None
        self.final = True

        logging.debug("%i vocab elements, including: %s" % (len(self), str(self.examples(10))))

    def examples(self, limit:int = 20) -> Iterable[str]:
        """
        Get examples from the vocab.  To make them interesting, sort by length.

        Args:
            limit: How many to return
        Returns:
            A list of the elements
        """
        return sorted(self.lookup, key=len, reverse=True)[:limit]
        
        
    def render(self, tokens: Iterable[int], separator:str=""):
      """
      Turn a token sequence into a string.

      Args:
        tokens: The integer token ids
        separator: Character to put between all tokens
      """
      result = []
      for token in tokens:
        token_string = self.lookup_word(token)
        assert token_string is not None, "Vocab lookup failed for %i" % token
        result.append(token_string)
      return separator.join(result)

    def save(self, filename:str) -> None:
        """
        Save the vocabulary to a file for easier inspection

        Args:
            filename: The file we create to save vocab to
        """
        with open(filename, 'w') as outfile:
            for ii in sorted(self._id_to_word):
                outfile.write("%i\t%s\n" % (ii, self._id_to_word[ii]))
                
    def set_default_index(self, idx):
        self.default_index = idx

    def __len__(self):
        return len(self.lookup)

    def __contains__(self, key):
        return key in self.lookup

    def __contains__(self, candidate: Union[int, str]) -> bool:
        """
        Check to see if something is part of the mapping

        Args:
            candidate: The element we're checking to see if it's in the mapping
        """
        if isinstance(candidate, str):
            return candidate in self.lookup
        elif isinstance(candidate, int):
            return candidate in self.reverse_lookup
        else:
            return False

    def __iter__(self) -> Iterable[Tuple[str, int]]:
        """
        Returns:
            Provides an iterator over all elements in the mapping
        """
        for key in sorted(self.lookup.keys()):
            yield key, self.lookup[key]
            
    def __getitem__(self, key):
        if key in self.lookup:
            return self.lookup[key]
        else:
            return self.default_index

    def lookup_token(self, word):
        return self.reverse_lookup[word]

    @staticmethod
    def string_from_bytes(self, bytestream: list[int], max_chars:int=1) -> str:
        """
        Given a bytetream, turn it into a string (if you can).

        Args:
            bytestream: Integers that may or may not represent a string
            max_chars: Do not return the string if it's longer than this many unicode characters
        Returns:
            UTF-8 representation of the bytestream if possible, None otherwise

        There's a more grounded version I did not use here:
        https://heycoach.in/blog/utf-8-validation-solution-in-python/
        """
        try:
            result = bytearray(bytestream).decode('utf-8')
        except UnicodeDecodeError:
            result = None

        if result and len(result) <= max_chars:
            return result
        else:
            return None
    
    @staticmethod
    def build_vocab_from_iterator(iterator, specials=[], max_tokens=-1, vocab_cutoff=2):
        from nltk.lm import Vocabulary
        from collections import Counter

        counts = Counter()
        for doc in iterator:
            counts.update(doc)

        return Vocab(counts, specials=specials, max_tokens=max_tokens, vocab_cutoff=vocab_cutoff)

if __name__ == "__main__":
    from guesser import kTOY_DATA
    from nltk.tokenize import word_tokenize

    tokenizer = word_tokenize

    vocab = Vocab.build_vocab_from_iterator([word_tokenize(doc["text"]) for doc in kTOY_DATA["mini-train"]],
                                            specials=["<unk>"], max_tokens=10)
    print("Unk:", vocab['<unk>'])
    vocab.set_default_index(vocab["<unk>"])
        
    for doc in kTOY_DATA["mini-dev"]:
        print(" ".join("%s_%i" % (x, vocab[x]) for x in word_tokenize(doc["text"])))
