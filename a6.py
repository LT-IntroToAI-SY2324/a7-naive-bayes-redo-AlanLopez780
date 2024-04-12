import math, os, pickle, re
from typing import Tuple, List, Dict


class BayesClassifier:
    """A simple BayesClassifier implementation

    Attributes:
        pos_freqs - dictionary of frequencies of positive words
        neg_freqs - dictionary of frequencies of negative words
        pos_filename - name of positive dictionary cache file
        neg_filename - name of positive dictionary cache file
        training_data_directory - relative path to training directory
        neg_file_prefix - prefix of negative reviews
        pos_file_prefix - prefix of positive reviews
    """

    def __init__(self):
        """Constructor initializes and trains the Naive Bayes Sentiment Classifier. If a
        cache of a trained classifier is stored in the current folder it is loaded,
        otherwise the system will proceed through training.  Once constructed the
        classifier is ready to classify input text."""
        # initialize attributes
        self.pos_freqs: Dict[str, int] = {}
        self.neg_freqs: Dict[str, int] = {}
        self.pos_filename: str = "pos.dat"
        self.neg_filename: str = "neg.dat"
        self.training_data_directory: str = "movie_reviews/"
        self.neg_file_prefix: str = "movies-1"
        self.pos_file_prefix: str = "movies-5"

        # check if both cached classifiers exist within the current directory
        if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename):
            print("Data files found - loading to use cached values...")
            self.pos_freqs = self.load_dict(self.pos_filename)
            self.neg_freqs = self.load_dict(self.neg_filename)
        else:
            print("Data files not found - running training...")
            self.train()

    def train(self) -> None:
        """Trains the Naive Bayes Sentiment Classifier

        Train here means generates `pos_freq/neg_freq` dictionaries with frequencies of
        words in corresponding positive/negative reviews
        """
        # get the list of file names from the training data directory
        # os.walk returns a generator (feel free to Google "python generators" if you're
        # curious to learn more, next gets the first value from this generator or the
        # provided default `(None, None, [])` if the generator has no values)
        _, __, files = next(os.walk(self.training_data_directory), (None, None, []))
        if not files:
            raise RuntimeError(f"Couldn't find path {self.training_data_directory}")

        # print(files)
        # print(len(files))
        # files now holds a list of the filenames
        # self.training_data_directory holds the folder name where these files are
        

        # stored below is how you would load a file with filename given by `fName`
        # fName = files[0]
        # print(fName)
        # `text` here will be the literal text of the file (i.e. what you would see
        # if you opened the file in a text editor
        # text = self.load_file(os.path.join(self.training_data_directory, fName))


        # *Tip:* training can take a while, to make it more transparent, we can use the
        # enumerate function, which loops over something and has an automatic counter.
        # write something like this to track progress (note the `# type: ignore` comment
        # which tells mypy we know better and it shouldn't complain at us on this line):
        for index, filename in enumerate(files, 1): # type: ignore
            print("------------------------------------------")
            print(f"Training {filename} on file {index} of {len(files)}")
        #     <the rest of your code for updating frequencies here>
            text = self.load_file(os.path.join(self.training_data_directory, filename))
            print(text)

        # we want to fill pos_freqs and neg_freqs with the correct counts of words from
        # their respective reviews
        
        # for each file, if it is a negative file, update (see the Updating frequencies
        # set of comments for what we mean by update) the frequencies in the negative
        # frequency dictionary. If it is a positive file, update (again see the Updating
        # frequencies set of comments for what we mean by update) the frequencies in the
        # positive frequency dictionary. If it is neither a postive or negative file,
        # ignore it and move to the next file (this is more just to be safe; we won't
        # test your code with neutral reviews)
        

        # Updating frequences: to update the frequencies for each file, you need to get
        # the text of the file, tokenize it, then update the appropriate dictionary for
        # those tokens. We've asked you to write a function `update_dict` that will make
        # your life easier here. Write that function first then pass it your list of
        # tokens from the file and the appropriate dictionary
            print(f"positive? {filename.startswith(self.pos_file_prefix)}")
            print(f"negative? {filename.startswith(self.neg_file_prefix)}")
            tokens = self.tokenize(text)
            print(tokens)
            
            if filename.startswith(self.pos_file_prefix):
                self.update_dict(tokens, self.pos_freqs)
            elif filename.startswith(self.neg_file_prefix):
                self.update_dict(tokens, self.neg_freqs)
        # for debugging purposes, it might be useful to print out the tokens and their
        # frequencies for both the positive and negative dictionaries
        
        # print(len(self.pos_freqs))
        # print(len(self.neg_freqs))
        # once you have gone through all the files, save the frequency dictionaries to
        # avoid extra work in the future (using the save_dict method). The objects you
        # are saving are self.pos_freqs and self.neg_freqs and the filepaths to save to
        # are self.pos_filename and self.neg_filename
        self.save_dict(self.pos_freqs, self.pos_filename)
        self.save_dict(self.neg_freqs, self.neg_filename)

    def classify(self, text: str) -> str:
        """Classifies given text as positive, negative or neutral from calculating the
        most likely document class to which the target string belongs

        Args:
            text - text to classify

        Returns:
            classification, either positive, negative or neutral
        """
        # TODO: fill me out

        
        # get a list of the individual tokens that occur in text
        tokens = self.tokenize(text)
        # print(tokens)

        # create some variables to store the positive and negative probability. since
        # we will be adding logs of probabilities, the initial values for the positive
        # and negative probabilities are set to 0
        pos_prob = 0
        neg_prob = 0

        # get the sum of all of the frequencies of the features in each document class
        # (i.e. how many words occurred in all documents for the given class) - this
        # will be used in calculating the probability of each document class given each
        # individual feature
        num_pos_words = sum(self.pos_freqs.values())
        num_neg_words = sum(self.neg_freqs.values())
        # print(num_pos_words)
        # print(num_neg_words)

        # for each token in the text, calculate the probability of it occurring in a
        # postive document and in a negative document and add the logs of those to the
        # running sums. when calculating the probabilities, always add 1 to the numerator
        # of each probability for add one smoothing (so that we never have a probability
        # of 0)
        for word in tokens:
            num_pos_appearances = 1
            if word in self.pos_freqs:
                num_pos_appearances += self.pos_freqs[word]
            # print(num_pos_appearances)
            pos_prob += math.log(num_pos_appearances / num_pos_words)

            num_neg_appearances = 1
            if word in self.neg_freqs:
                num_neg_appearances += self.neg_freqs[word]
            # print(num_neg_appearances)
            neg_prob += math.log(num_neg_appearances / num_neg_words)

        # for debugging purposes, it may help to print the overall positive and negative
        # probabilities
        print(f"Positive Probability: {pos_prob}")
        print(f"Negative Probability: {neg_prob}")

        # determine whether positive or negative was more probable (i.e. which one was
        # larger)
        if pos_prob > neg_prob:
            return "positive"
        else:
            return "negative"

        # return a string of "positive" or "negative"

    def load_file(self, filepath: str) -> str:
        """Loads text of given file

        Args:
            filepath - relative path to file to load

        Returns:
            text of the given file
        """
        with open(filepath, "r", encoding='utf8') as f:
            return f.read()

    def save_dict(self, dict: Dict, filepath: str) -> None:
        """Pickles given dictionary to a file with the given name

        Args:
            dict - a dictionary to pickle
            filepath - relative path to file to save
        """
        print(f"Dictionary saved to file: {filepath}")
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)

    def load_dict(self, filepath: str) -> Dict:
        """Loads pickled dictionary stored in given file

        Args:
            filepath - relative path to file to load

        Returns:
            dictionary stored in given file
        """
        print(f"Loading dictionary from file: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()

    def tokenize(self, text: str) -> List[str]:
        """Splits given text into a list of the individual tokens in order

        Args:
            text - text to tokenize

        Returns:
            tokens of given text in order
        """
        tokens = []
        token = ""
        for c in text:
            if (
                re.match("[a-zA-Z0-9]", str(c)) != None
                or c == "'"
                or c == "_"
                or c == "-"
            ):
                token += c
            else:
                if token != "":
                    tokens.append(token.lower())
                    token = ""
                if c.strip() != "":
                    tokens.append(str(c.strip()))

        if token != "":
            tokens.append(token.lower())
        return tokens

    def update_dict(self, words: List[str], freqs: Dict[str, int]) -> None:
        """Updates given (word -> frequency) dictionary with given words list

        By updating we mean increment the count of each word in words in the dictionary.
        If any word in words is not currently in the dictionary add it with a count of 1.
        (if a word is in words multiple times you'll increment it as many times
        as it appears)

        Args:
            words - list of tokens to update frequencies of
            freqs - dictionary of frequencies to update
        """
        # TODO: your work here
        for word in words:
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1


if __name__ == "__main__":
    # uncomment the below lines once you've implemented `train` & `classify`
    b = BayesClassifier()
    a_list_of_words = ["I", "really", "like", "this", "movie", ".", "I", "hope", \
                       "you", "like", "it", "too"]
    a_dictionary = {}
    b.update_dict(a_list_of_words, a_dictionary)
    assert a_dictionary["I"] == 2, "update_dict test 1"
    assert a_dictionary["like"] == 2, "update_dict test 2"
    assert a_dictionary["really"] == 1, "update_dict test 3"
    assert a_dictionary["too"] == 1, "update_dict test 4"
    print("update_dict tests passed.")

    pos_denominator = sum(b.pos_freqs.values())
    neg_denominator = sum(b.neg_freqs.values())

    print("\nThese are the sums of values in the positive and negative dicitionaries.")
    print(f"sum of positive word counts is: {pos_denominator}")
    print(f"sum of negative word counts is: {neg_denominator}")

    print("\nHere are some sample word counts in the positive and negative dicitionaries.")
    print(f"count for the word 'love' in positive dictionary {b.pos_freqs['love']}")
    print(f"count for the word 'love' in negative dictionary {b.neg_freqs['love']}")
    print(f"count for the word 'terrible' in positive dictionary {b.pos_freqs['terrible']}")
    print(f"count for the word 'terrible' in negative dictionary {b.neg_freqs['terrible']}")
    print(f"count for the word 'computer' in positive dictionary {b.pos_freqs['computer']}")
    print(f"count for the word 'computer' in negative dictionary {b.neg_freqs['computer']}")
    print(f"count for the word 'science' in positive dictionary {b.pos_freqs['science']}")
    print(f"count for the word 'science' in negative dictionary {b.neg_freqs['science']}")
    print(f"count for the word 'i' in positive dictionary {b.pos_freqs['i']}")
    print(f"count for the word 'i' in negative dictionary {b.neg_freqs['i']}")
    print(f"count for the word 'is' in positive dictionary {b.pos_freqs['is']}")
    print(f"count for the word 'is' in negative dictionary {b.neg_freqs['is']}")
    print(f"count for the word 'the' in positive dictionary {b.pos_freqs['the']}")
    print(f"count for the word 'the' in negative dictionary {b.neg_freqs['the']}")

    print("\nHere are some sample probabilities.")
    print(f"P('love'| pos) {(b.pos_freqs['love']+1)/pos_denominator}")
    print(f"P('love'| neg) {(b.neg_freqs['love']+1)/neg_denominator}")
    print(f"P('terrible'| pos) {(b.pos_freqs['terrible']+1)/pos_denominator}")
    print(f"P('terrible'| neg) {(b.neg_freqs['terrible']+1)/neg_denominator}")

    # # uncomment the below lines once you've implemented `classify`
    print("\nThe following should all be positive.")
    print(b.classify('I love computer science'))
    print(b.classify('this movie is fantastic'))
    print("\nThe following should all be negative.")
    print(b.classify('rainy days are the worst'))
    print(b.classify('computer science is terrible'))

    print("\nThe following is to test out the method with each groups responses")
    print(b.classify("I'm so excited for the solar eclipse! It's going to be so cool!"))
    print(b.classify("This day has been going good so far. Hopefully it gets better later."))
    print(b.classify("That guy over there seems to be very kind and friendly. I think you could be friends with him."))
    print(b.classify("This movie is really great! I recommend anyone to watch it since it’s very creative and interesting."))
    print(b.classify("The solar eclipse is going to be boring, why should we waste our time?"))
    print(b.classify("Today has been the absolute worst! And it better not get even more worse later."))
    print(b.classify("That guy over there looks very mean and he could also be a bully too. Don’t bother even becoming friends with him."))
    print(b.classify("This movie is absolutely atrocious! The plot is boring and there’s too much swearing and violence as well because I’m an immature little piece of shit!"))
    print(b.classify("A classic film. John Carpenter's The Thing is one of the most entertaining horror films ever made - fast, clever and purely exciting from start to finish. It is one of my personal favorite horror movies. This is how all movies of the genre should be made. Set on an isolated base in Antarctica, this version seems almost to pick up where the original version (The Thing From Another World) left off. The American scientists discover a decimated Norwegian base some miles distant. Everyone is dead, and only the half charred remains of some unidentifiable thing left to smolder outside the compound might offer any answers to what may have happened. The Thing is brought back to the American base and, too late, the scientists realize that it is alive and lethal. The Thing thaws out and is off, not only killing anyone and anything that crosses Its path, but also absorbing them, making Itself into whoever and whatever it wants. The film then turns into a brilliant paranoia piece. Everyone is suspect, anyone can be The Thing, and no one trusts anyone anymore. Gone is the strength and security found when human beings band together in spite of their differences to battle a monster. The group splinters and fear rules supreme. Who is the Thing? Seriously I Love this movie I love it To Death. I love Escape From New York and I love Escape From L.A. but I also love The Thing so much better this is definitely the best Carpenter film a truly masterpiece classic I love R.J. MacReady - Kurt Russell I love everything about this film that is. Science Fiction, Horror and an Action Epic Film. A lot of the practical effects were left out but the it looked nice and the acting was good and it expanded upon the monsters background and showing you the inside of the ship. It must of been tough to bring across on screen the visual design. In my opinion, nobody has topped this film in the 25-odd years since its release. I'll put any of The Thing's old-school effects up against any CGI-driven movie, or this cast against almost any other ensemble. If you haven't seen the film yet, I envy you because I WISH I could see The Thing again for the first time. WOW! Does more need to be said? How about this...there is no parallel. Who's your friend? Who's the Thing? Who do you trust? Who can you afford to trust? If you've never seen this movie...your in for a treat. The only other movie that had such an impact on me was The Matrix (the first movie)...where I left the theater touching the walls wondering if they were really real. This movie will leaving you wondering....is the guy/gal next to you really real? Trust is a tough thing to come by these days. John Carpenter's The Thing is a seminal piece of horror that is not only a fine specimen of its era, but it also serves as a shining example of horror done absolutely right in any era. Combining gross-out special effects reminiscent of Hellraiser, the nail-bitingly intense, claustrophobic filmmaking of Alien, offering a story that is very well-paced, such as George A. Romero's Dawn of the Dead, and presenting the idea that true terror can be found at any time, in any place, and inside anyone, much like Alfred Hitchcock's Psycho, The Thing works on every level, and represents the peak of each and every aspect that may be utilized to make horror films effective. Granted, this amalgamation of styles is not the only formula for winning horror. The basic plot about this movie is Horror-moister John Carpenter (Halloween, Escape from New York) teams Kurt Russell's outstanding performance with incredible visuals to build this chilling version of the classic The Thing. In the winter of 1982, a twelve-man research team at a remote Antarctic research station discovers an alien buried in the snow for over 100,000 years. Once unfrozen, the form-changing alien wreaks havoc, creates terror and becomes one of them. It is one of the best favorite horror films of the 80's ever. I love this film to death. The Thing is the best classic horror film from master and genius John Carpenter! 'The Thing' is classic Carpenter and one of the few remakes that is better than the original. Kurt Russell's characters: Snake Plissken, R.J. MacReady and Jack Burton are Kurt's best favorite characters he ever played. I also love the music score from Ennio Morricone! Awesome!!! 10/10 Grade: Bad Ass Seal Of Approval"))
    print(b.classify("'American Psycho' is NOT a slasher movie. It is a depiction, a fantasy if you will, of the life of modern man and his place in society. Nothing is enough. Money, sex, social stature, there is always someone else who has more and everyone else expect from you to try harder for even more. This movie is about eliminating competition the easy way. By killing your opponents. By eating your sexual partners. By destroying everyone around you. 'American Psycho' retains the balance between this psychotic state, a chilling thriller and a very funny movie. The scenes that show Patrick playing music for his guests are absolutely hilarious, as he comments very seriously on records by artists such as Whitney Houston, Phil Collins and Huey Lewis & the News. The funny thing is that he chooses the most commercial or sold out records of these artists, to explain how much better they are compared to their previous, more artistic work. Another message of the state of the receivers of commercial art. You can analyze 'American Psycho' for hours. It can be perceived both as a deep and a fun movie. Even if you don't like the story, you will love Christian Bale's excellent performance. Enjoy."))
    print(b.classify("The idea was way too simple, just an angry and ruthless shark swimming around and eating defenceless people. When I was 6 I was really scared of it, bloody water and all those documentaries about sharks attacking people. But now I grew up and probably also grew out of flicks like this. It isn't even so funny now. It's just dull, boring and unwatchable. I don't know how a thinking person call it the best horror. It's just a clichéd thriller with some major flaws. Watch Alien - it's a true horror."))
    print(b.classify("Okay, so bad acting, poor plot, cheap effects and cheesy photography are what make "the greatest horror movie of all time"? My 15 year old son said, "Well, maybe they liked it, because it's old". Yeah, either that, or Sam Raimi paid dozens of friends and family to vote for this clunker!!! Hey, it isn't a horrible movie: just dumb. I suppose if you're drunk, and have nothing better to do on a Saturday night, then rent this movie. Oh, in one of the "scary" moments of the film, my son said, "These people are dumb... they're retarded". The same might be said for anyone who actually believes this cinematic garbage is "essential viewing for a movie buff." (For "essential viewing, try "The Maltese Falcon" or "Amelie" or a film by Fellini or Bergman, if you want essential viewing... but NOT this sad waste of film!

One final note: the fact that so many folks on IMDb rated this film "10 stars" should be a red flag (suspicious sign, for those of you who have never heard the expression)-- Evidently, the average reviewer on this site is 1. uneducated 2. unsophisticated 3. under the age of sixteen 4. never seen a truly great film or 5. is a shill for the movie industry. I mean, ten stars should be reserved for a film in the top 100 greatest movies: not this sad little excuse for a story."))
    pass