from gensim.models.word2vec import BrownCorpus, Word2Vec, Vocab, logger
from os.path import isfile as file_exists
import re, os, gzip, pickle, numpy as np, theano, threading, time
try:
	from queue import Queue
except ImportError:
	from Queue import Queue
from gensim import utils, matutils
from numpy import zeros, empty, float32 as REAL, argsort, dot, random
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from .word2vec_inner import train_sentence_sg_original, train_sentence_sg_svrg, train_sentence_sg_double_svrg

UnknownWord = "**UNKNOWN**"
UnknownUppercaseWord = "**UNKNOWN_CAP**"

class BrownCorpusSimple(BrownCorpus):
	"""Iterate over sentences from the Brown corpus (part of NLTK data)."""
	def __init__(self, fname):
		self.fname = fname

	def __iter__(self):
		if os.path.isdir(self.fname):
			for fname in os.listdir(self.fname):
				fname = os.path.join(self.fname, fname)
				if not os.path.isfile(fname):
					continue
				for line in open(fname):
					# each file line is a single sentence in the Brown corpus
					# each token is WORD/POS_TAG
					words = [t.split('/')[0].lower() for t in line.split() if len(t.split('/')) == 2]
					# ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
					if not words:  # don't bother sending out empty sentences
						continue
					yield words
		else:
			for line in open(self.fname):
				print(1)
				# each file line is a single sentence in the Brown corpus
				# each token is WORD/POS_TAG
				words = [t.split('/')[0].lower() for t in line.split() if len(t.split('/')) == 2]
				# ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
				if not words:  # don't bother sending out empty sentences
					continue
				yield words

bad_liner = re.compile("(\||[&=][a-zA-Z])")

class LineCorpus(object):
	def __init__(self, fname, filter_lines = True):
		"""Simple format: one sentence = one line; words already preprocessed and separated by whitespace.

		fname can be either a string or a file object

		Thus, one can use this for just plain files:

			sentences = LineSentence('myfile.txt')

		Or for compressed files:

			sentences = LineSentence(bz2.BZ2File('compressed_text.bz2'))
		"""
		self.fname = fname
		self.filter_lines = filter_lines

	def good_line(self, line):
		if not self.filter_lines: return True
		else: return not bad_liner.search(line)

	def __iter__(self):
		"""Iterate through the lines in the fname."""
		try:
			# Assume it is a file-like object and try treating it as such
			# Things that don't have seek will trigger an exception
			self.fname.seek(0)
			for line in self.fname:
				yield line.split()
		except AttributeError as e:
			print("AttributeError => ", str(e))
			# If it didn't work like a file, use it as a string filename
			with utils.smart_open(self.fname, "rb") as fin:
				for line in fin:
					line = line.decode("utf-8")
					if self.good_line(line):
						yield line.split()

def create_corpus_from_matlab(word_embedding, index2word):
	model            = Word2VecExtended()
	model.syn0       = word_embedding.astype(theano.config.floatX).copy()
	model.index2word = index2word
	model.index2word[0] = UnknownWord
	vocab = {}

	for word in model.index2word:
		v = Vocab(count=1)
		v.index = len(vocab)
		vocab[word] = v

	model.vocab      = vocab
	model.UnknownWordIndex = model.vocab[UnknownWord].index
	return model

class Word2VecExtended(Word2Vec):
	"""
	Word2Vec trainer based off of gensim's word2vec
	implementation.
	"""

	def __init__(self,
			sentences=None,
			oov_word = False,
			size=100,
			alpha=0.035,
			window=5,
			min_count=5,
			seed=1,
			workers=1,
			min_alpha=0.0001,
			sg=1,
			training_function = train_sentence_sg_original,
			decay = True,
			vocab_report_frequency = 10000):
		"""
		Construct a word2vec language model from a corpus `sentences`.
		Optionally specify whether an out of vocabulary word should be
		trained.

		Inputs (optional keyword arguments)
		-----------------------------------

		oov_word              bool : train unknown word vector?
		size                   int : embedding dimensions.
		alpha                float : learning rate
		decay                 bool : anneal learning rate over time?
		min_alpha            float : if learning rate is annealed, minimum rate to use.
		sg                     int : use skip gram method or average context?
		training_function function : training function that takes as parameters:
			Word2VecExtended, list<int>, float, np.array (stores gradient)
		window                 int : size of the context used to train embeddings.
		min_count              int : minimum word occurence to include in model vocabulary.
		seed                   int : random seed to set up weights
		sentences           object : what corpus to train on (See: `LineCorpus`,
			`BrownCorpus`, `BrownCorpusSimple`)

		"""

		self.vocab = {}  # mapping from a word (string) to a Vocab object
		self.index2word = []  # map from a word's matrix index (int) to word (string)
		self.sg = int(sg)
		self.layer1_size = int(size)
		self.logistic_regression_size = self.layer1_size
		if size % 4 != 0:
			logger.warning("consider setting layer size to a multiple of 4 for greater performance")
		self.alpha = float(alpha)
		self.window = int(window)
		self.weight_decay = decay
		self.seed = seed
		self.hs = True
		self.negative = False
		self.training_function = training_function
		self.min_count = min_count
		self.workers = workers
		self.min_alpha = min_alpha

		if sentences is not None:
			self.build_vocab(sentences, oov_word = oov_word, report_frequency = vocab_report_frequency)
			self.train(sentences) # maybe ?

	def accuracy(self, questions, restrict_vocab=30000):
		"""
		Compute accuracy of the model (with **capitalizations**). `questions` is a filename where lines are
		4-tuples of words, split into sections by ": SECTION NAME" lines.
		See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

		The accuracy is reported (=printed to log and returned as a list) for each
		section separately, plus there's one aggregate summary at the end.

		Use `restrict_vocab` to ignore all questions containing a word whose frequency
		is not in the top-N most frequent words (default top 30,000).

		This method corresponds to the `compute-accuracy` script of the original C word2vec.

		"""
		ok_vocab = dict(sorted(self.vocab.items(),
							   key=lambda item: -item[1].count)[:restrict_vocab])
		ok_index = set(v.index for v in ok_vocab.values())

		def log_accuracy(section):
			correct, incorrect = section['correct'], section['incorrect']
			if correct + incorrect > 0:
				logger.info("%s: %.1f%% (%i/%i)" %
					(section['section'], 100.0 * correct / (correct + incorrect),
					correct, correct + incorrect))

		sections, section = [], None
		for line_no, line in enumerate(open(questions)):
			# TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
			if line.startswith(': '):
				# a new section starts => store the old section
				if section:
					sections.append(section)
					log_accuracy(section)
				section = {'section': line.lstrip(': ').strip(), 'correct': 0, 'incorrect': 0}
			else:
				if not section:
					raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
				try:
					a, b, c, expected = line.split()  # TODO assumes vocabulary preprocessing uses lowercase, too...
				except:
					logger.info("skipping invalid line #%i in %s" % (line_no, questions))
				if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
					logger.debug("skipping line #%i with OOV words: %s" % (line_no, line))
					continue

				ignore = set(self.vocab[v].index for v in [a, b, c])  # indexes of words to ignore
				predicted = None
				# find the most likely prediction, ignoring OOV words and input words
				for index in argsort(self.most_similar(positive=[b, c], negative=[a], topn=False))[::-1]:
					if index in ok_index and index not in ignore:
						predicted = self.index2word[index]
						if predicted != expected and predicted != expected.lower():
							logger.debug("%s: expected %s, predicted %s" % (line.strip(), expected, predicted))
						break
				section['correct' if predicted == expected else 'incorrect'] += 1
		if section:
			# store the last section, too
			sections.append(section)
			log_accuracy(section)

		total = {'section': 'total', 'correct': sum(s['correct'] for s in sections), 'incorrect': sum(s['incorrect'] for s in sections)}
		log_accuracy(total)
		sections.append(total)
		return sections

	def extend_vocab(self, sentences, oov_word = False, report_frequency = 10000):
		"""
		Extend vocabulary from a sequence of sentences (can be a once-only generator stream).
		Each sentence must be a list of utf8 strings.

		"""
		logger.info("collecting all words and their counts")

		prev_sentence_no = -1
		sentence_no, vocab = -1, {}
		total_words = 0
		assign_to_vocab = vocab.__setitem__ # slight performance gain
		# https://wiki.python.org/moin/PythonSpeed/PerformanceTips
		get_from_vocab = vocab.__getitem__
		for sentence_no, sentence in enumerate(sentences):
			if prev_sentence_no == sentence_no:
				break
			if sentence_no % report_frequency == 0:
				logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
					(sentence_no, total_words, len(vocab)))
			for word in sentence:
				if word in vocab:
					get_from_vocab(word).count += 1
				else:
					assign_to_vocab(word, Vocab(count=1))
			total_words += len(sentence)
			prev_sentence_no = sentence_no
		logger.info("collected %i word types from a corpus of %i words and %i sentences" %
			(len(vocab), total_words, sentence_no + 1))

		# assign a unique index to each word
		append = self.index2word.append
		assign_to_vocab = self.vocab.__setitem__
		for word, v in vocab.items():
			if word not in self.vocab:
				if v.count >= self.min_count:
					v.index = len(self.vocab)
					append(word)
					assign_to_vocab(word, v)
			else:
				self.vocab[word].count += v.count

		# add the special out of vocabulary word **UNKNOWN**:
		if oov_word:
			self.add_oov_word(count = len(vocab) - len(self.vocab))

		logger.info("total %i word types after removing those with count<%s" % (len(self.vocab), self.min_count))

		# add info about each word's Huffman encoding
		self.create_binary_tree()
		self.extend_weights()

	def extend_weights(self):
		"""Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
		logger.info("resetting layer weights")
		random.seed(self.seed)
		old_size = self.syn0.shape[0]
		self.syn0 = np.vstack([self.syn0, empty((len(self.vocab) - old_size, self.layer1_size), dtype=REAL)])

		# randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
		for i in range(old_size, len(self.vocab)):
			self.syn0[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
		self.syn1 = zeros((len(self.vocab), self.logistic_regression_size), dtype=REAL)
		self.syn0norm = None

	def build_vocab(self, sentences, oov_word = False, report_frequency = 10000):
		"""
		Build vocabulary from a sequence of sentences (can be a once-only generator stream).
		Each sentence must be a list of utf8 strings.

		"""
		print("build vocab")
		path = (re.sub("/","_",sentences.fname)+ ("(mc=%d)" % (self.min_count)) + ".vocab") if hasattr(sentences, "fname") else None
		if path != None and file_exists(path):
			logger.info("loading from saved vocab list at \"%s\"" % (path))
			file = gzip.open(path, 'r')
			saved_vocab = pickle.load(file)
			file.close()
			self.index2word = saved_vocab["index2word"]
			self.vocab      = saved_vocab["vocab"]

			if oov_word:
				self.add_oov_word(count = 10000)

			self.create_binary_tree()
			self.reset_weights()



		else:
			logger.info("collecting all words and their counts")

			prev_sentence_no = -1
			sentence_no, vocab = -1, {}
			total_words = 0
			assign_to_vocab = vocab.__setitem__ # slight performance gain
			# https://wiki.python.org/moin/PythonSpeed/PerformanceTips
			get_from_vocab = vocab.__getitem__
			for sentence_no, sentence in enumerate(sentences):
				if prev_sentence_no == sentence_no:
					break
				if sentence_no % report_frequency == 0:
					logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
						(sentence_no, total_words, len(vocab)))
				for word in sentence:
					if word in vocab:
						get_from_vocab(word).count += 1
					else:
						assign_to_vocab(word, Vocab(count=1))
				total_words += len(sentence)
				prev_sentence_no = sentence_no
			logger.info("collected %i word types from a corpus of %i words and %i sentences" %
				(len(vocab), total_words, sentence_no + 1))

			# assign a unique index to each word
			self.vocab, self.index2word = {}, []
			append = self.index2word.append
			assign_to_vocab = self.vocab.__setitem__
			for word, v in vocab.items():
				if v.count >= self.min_count:
					v.index = len(self.vocab)
					append(word)
					assign_to_vocab(word, v)

			# add the special out of vocabulary word **UNKNOWN**:
			if oov_word:
				self.add_oov_word(count = len(vocab) - len(self.vocab))
			len(vocab) - len(self.vocab)

			logger.info("total %i word types after removing those with count<%s" % (len(self.vocab), self.min_count))

			# add info about each word's Huffman encoding
			self.create_binary_tree()
			self.reset_weights()
			if path != None:
				logger.info("saving vocab list in \"%s\"" % (path))
				with gzip.open(path, 'wb') as file:
					pickle.dump({"vocab": self.vocab, "index2word": self.index2word}, file, 1)
	def add_oov_word(self, count = 1):
		if UnknownWord not in self.vocab:
			v = self.add_word_to_vocab(UnknownWord, count = count)
			self.UnknownWordIndex = v.index
		else:
			self.UnknownWordIndex = self.vocab[UnknownWord].index
		if UnknownUppercaseWord not in self.vocab:
			v2 = self.add_word_to_vocab(UnknownUppercaseWord, count = count)
			self.UnknownUppercaseWordIndex = v2.index
		else:
			self.UnknownUppercaseWordIndex = self.vocab[UnknownUppercaseWord].index


	def add_word_to_vocab(self, word, count = 1):
		v = Vocab(count = count)
		v.index = len(self.vocab)
		self.vocab[word] = v
		self.index2word.append(word)
		return v

	def get_underlying_word(self, word):
		ind = self.vocab.get(word, None)
		if ind is not None:
			return word
		else:
			return UnknownWord

	def search_vector(self, vector,  topn=10):
		mean = vector
		mean = matutils.unitvec(mean).astype(REAL)
		dists = dot(self.syn0norm, mean)
		if not topn:
			return dists
		best = argsort(dists)[::-1][:topn]
		# ignore (don't return) words from the input
		result = [(self.index2word[sim], float(dists[sim])) for sim in best]
		return result[:topn]

	def get_underlying_word_object(self, word):
		ind = self.vocab.get(word, None)
		if ind is not None:
			return ind
		else:
			return self.vocab.get(UnknownWord, None)

	def get_index(self, word):
		"""
		Return the index of the word vector associated to a word, or
		return the index for the unknown (out of vocabulary) word
		vector.

		"""
		ind = self.vocab.get(word, None)
		if ind != None:
			return ind.index
		else:
			return self.UnknownWordIndex

	def get(self, word):
		"""
		Return the word vector associated to a word, or
		return the unknown (out of vocabulary) word vector

		"""
		return self.syn0[self.get_index(word)]

	def reset_weights(self):
		"""Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
		logger.info("resetting layer weights")
		random.seed(self.seed)
		self.syn0 = empty((len(self.vocab), self.layer1_size), dtype=REAL)
		# randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
		for i in range(len(self.vocab)):
			self.syn0[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
		self.syn1 = zeros((len(self.vocab), self.logistic_regression_size), dtype=REAL)
		self.syn0norm = None

	def train(self, sentences, total_words=None, word_count=0, chunksize=100):
		"""
		Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
		Each sentence must be a list of utf8 strings.

		"""
		# oov_replacement = self.vocab.get(UnknownWord,None)

		if not self.vocab:
			raise RuntimeError("you must first build vocabulary before training the model")

		start, next_report = time.time(), [1.0]
		word_count, total_words = [word_count], total_words or sum(v.count for v in self.vocab.values())
		jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
		lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

		def worker_train():
			"""Train the model, lifting lists of sentences from the jobs queue."""
			work = zeros(self.layer1_size, dtype=REAL)  # each thread must have its own work memory
			neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
			while True:
				job = jobs.get()
				if job is None:  # data finished, exit
					break
				# update the learning rate before every job
				alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words)) if self.weight_decay else self.alpha
				# how many words did we train on? out-of-vocabulary (unknown) words do not count
				job_words          = 0

				for sentence in job:
					job_words  += self.training_function(self, sentence, alpha, work)

				with lock:
					# here we can store the scores for later plotting and viewing...
					word_count[0] += job_words

					elapsed = time.time() - start
					if elapsed >= next_report[0]:
						logger.debug("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %
							(100.0 * word_count[0] / total_words, alpha, word_count[0] / elapsed if elapsed else 0.0))
						next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

		workers = [threading.Thread(target=worker_train) for _ in range(self.workers)]
		for thread in workers:
			thread.daemon = True  # make interrupting the process with ctrl+c easier
			thread.start()

		# convert input strings to Vocab objects (or None for OOV words), and start filling the jobs queue
		no_oov = ([self.get_underlying_word_object(word) for word in sentence] for sentence in sentences)
		for job_no, job in enumerate(utils.grouper(no_oov, chunksize)):
			# logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
			jobs.put(job)
		logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
		for _ in range(self.workers):
			jobs.put(None)  # give the workers heads up that they can finish -- no more work!

		for thread in workers:
			thread.join()

		elapsed = time.time() - start
		logger.info("training on %i words took %.1fs, %.0f words/s" %
			(word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))

		return word_count[0]

__all__ = [""]

