import fnmatch
import os
import random
import re
import threading

import librosa
import numpy as np
import tensorflow as tf

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename, category_id


# def load_vctk_audio(directory, sample_rate):
#     '''Generator that yields audio waveforms from the VCTK dataset, and
#     additionally the ID of the corresponding speaker.'''
#     files = find_files(directory)
#     speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
#     for filename in files:
#         audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
#         audio = audio.reshape(-1, 1)
#         matches = speaker_re.findall(filename)[0]
#         speaker_id, recording_id = [int(id_) for id_ in matches]
#
#         dirs_vctk_wav48_p, name = os.path.split(filename)
#         dirs_vctk_wav48, p = os.path.split(dirs_vctk_wav48_p)
#         dirs_vctk, wav48 = os.path.split(dirs_vctk_wav48)
#         filename_text = os.path.join(dirs_vctk, 'txt', p, name[:-4] + '.txt')
#
#         with open(filename_text) as f:
#             text = f.read()
#         yield audio, (filename, speaker_id, list(text))


def original_to_generated(path, ya_gen_path):
    pxxx_dir, wavname = os.path.split(path)
    base_dir, pxxx = os.path.split(pxxx_dir)
    return os.path.join(ya_gen_path, pxxx, wavname)


def fix_ya_len(audio, ya_audio):
    len_orig = len(audio)
    len_ya = len(ya_audio)
    if len_orig < len_ya:
        ya_audio = ya_audio[:len_orig]
    if len_orig > len_ya:
        ya_audio += audio[len_ya:]

    return ya_audio


def read_matching_ya(wav_path, sr=16000):
    ya_path = original_to_generated(wav_path)

    audio, _ = librosa.load(wav_path, sr=sr)
    ya_audio, _ = librosa.load(ya_path, sr=sr)
    len_ratio = len(audio) / len(ya_audio)
    new_sr = sr * len_ratio
    ya_audio, _ = librosa.load(ya_path, sr=new_sr)

    len_orig = len(audio)
    len_ya = len(ya_audio)
    if len_orig < len_ya:
        ya_audio = ya_audio[:len_orig]
    if len_orig > len_ya:
        ya_audio += audio[len_ya:]

    return fix_ya_len(audio, ya_audio)


# TODO read both audio and ya_audio and yeld them from iterator
def load_vctk_audio_ya(directory, sample_rate, ya_gen_path):
    '''Generator that yields audio waveforms from the VCTK dataset, and
    additionally the ID of the corresponding speaker.'''
    files = find_files(directory)
    # ya_files = [original_to_generated(f, ya_gen_path) for f in files]

    speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
    for filename, ya_filename in zip(files, ya_files):
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        matches = speaker_re.findall(filename)[0]
        speaker_id, recording_id = [int(id_) for id_ in matches]

        yield audio, (filename, speaker_id, ya_filename)


def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 gc_enabled,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32,
                 vctk=False,
                 lc_enabled=False,
                 vctk_ya=False):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.lc_enabled = lc_enabled
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        ###
        self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())

        self.lc_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.conditions_queue = tf.PaddingFIFOQueue(queue_size,
                                                    ['float32', 'string'],
                                                    shapes=[(), (None,)])

        # self.conditions_queue = tf.PaddingFIFOQueue(queue_size,
        #                                             ['int32', 'string'],
        #                                             shapes=[(), (None,)])

        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        self.conditions_enqueue = self.conditions_queue.enqueue([
            self.id_placeholder,
            self.lc_placeholder])
        ###

        # self.enqueue = self.queue.enqueue([self.sample_placeholder])
        # self.vctk = vctk
        self.vctk_ya = vctk_ya

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        if self.lc_enabled:
            self.lc_placeholder = tf.placeholder(dtype=tf.int32, shape=1)

        # self.lc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
        #                                     shapes=[()])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    # TODO return batch alongside with local condition feature batch
    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            if self.vctk_ya:
                iterator = load_vctk_audio_ya(self.audio_dir, self.sample_rate)
            else:
                iterator = load_generic_audio(self.audio_dir, self.sample_rate)
            for audio, extra in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio = trim_silence(audio[:, 0], self.silence_threshold)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(extra))

                if self.sample_size:
                    # Cut samples into fixed size pieces
                    buffer_ = np.append(buffer_, audio)
                    while len(buffer_) > self.sample_size:
                        piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        buffer_ = buffer_[self.sample_size:]
                        if self.vctk_ya:
                            filename, user_id, text = extra
                            sess.run(self.conditions_enqueue,
                                     feed_dict={self.id_placeholder: user_id,
                                                self.lc_placeholder: text})
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: np.reshape(audio, (-1, 1))})
                    if self.vctk_ya:
                        filename, speaker_id, ya_filename = extra
                        # filename, user_id, text = extra
                        sess.run(self.conditions_enqueue,
                                 feed_dict={self.id_placeholder: speaker_id,
                                            self.lc_placeholder: text})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
