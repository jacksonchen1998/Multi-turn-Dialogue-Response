DATA_FILES = lambda data_dir: {
    "train": [
        f"{data_dir}/sys_dialog_texts.train.npy",
        f"{data_dir}/sys_target_texts.train.npy",
        f"{data_dir}/sys_emotion_texts.train.npy",
        f"{data_dir}/sys_situation_texts.train.npy",
    ],
    "dev": [
        f"{data_dir}/sys_dialog_texts.dev.npy",
        f"{data_dir}/sys_target_texts.dev.npy",
        f"{data_dir}/sys_emotion_texts.dev.npy",
        f"{data_dir}/sys_situation_texts.dev.npy",
    ],
    "test": [
        f"{data_dir}/sys_dialog_texts.test.npy",
        f"{data_dir}/sys_target_texts.test.npy",
        f"{data_dir}/sys_emotion_texts.test.npy",
        f"{data_dir}/sys_situation_texts.test.npy",
    ],
}
START_TAG = "<START>"
STOP_TAG = "<STOP>"
WORD_PAIRS = {
    "it's": "it is",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "you'd": "you would",
    "you're": "you are",
    "you'll": "you will",
    "i'm": "i am",
    "they're": "they are",
    "that's": "that is",
    "what's": "what is",
    "couldn't": "could not",
    "i've": "i have",
    "we've": "we have",
    "can't": "cannot",
    "i'd": "i would",
    "i'd": "i would",
    "aren't": "are not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "there's": "there is",
    "there're": "there are",
}

EMO_MAP = {
    "surprised": 0,
    "excited": 1,
    "annoyed": 2,
    "proud": 3,
    "angry": 4,
    "sad": 5,
    "grateful": 6,
    "lonely": 7,
    "impressed": 8,
    "afraid": 9,
    "disgusted": 10,
    "confident": 11,
    "terrified": 12,
    "hopeful": 13,
    "anxious": 14,
    "disappointed": 15,
    "joyful": 16,
    "prepared": 17,
    "guilty": 18,
    "furious": 19,
    "nostalgic": 20,
    "jealous": 21,
    "anticipating": 22,
    "embarrassed": 23,
    "content": 24,
    "devastated": 25,
    "sentimental": 26,
    "caring": 27,
    "trusting": 28,
    "ashamed": 29,
    "apprehensive": 30,
    "faithful": 31,
    # "neutral": 32
    "questioning": 32,
    "acknowledging": 33,
    "agreeing": 34,
    "consoling": 35,
    "encouraging": 36,
    "sympathizing": 37,
    "wishing": 38,
    "suggesting": 39,
    "neutral": 40
}

MAP_DIA_EMO = {
    0: "surprised",
    1: "excited",
    2: "annoyed",
    3: "proud",
    4: "angry",
    5: "sad",
    6: "grateful",
    7: "lonely",
    8: "impressed",
    9: "afraid",
    10: "disgusted",
    11: "confident",
    12: "terrified",
    13: "hopeful",
    14: "anxious",
    15: "disappointed",
    16: "joyful",
    17: "prepared",
    18: "guilty",
    19: "furious",
    20: "nostalgic",
    21: "jealous",
    22: "anticipating",
    23: "embarrassed",
    24: "content",
    25: "devastated",
    26: "sentimental",
    27: "caring",
    28: "trusting",
    29: "ashamed",
    30: "apprehensive",
    31: "faithful",
}

MAP_EMO = {
    0: "surprised",
    1: "excited",
    2: "annoyed",
    3: "proud",
    4: "angry",
    5: "sad",
    6: "grateful",
    7: "lonely",
    8: "impressed",
    9: "afraid",
    10: "disgusted",
    11: "confident",
    12: "terrified",
    13: "hopeful",
    14: "anxious",
    15: "disappointed",
    16: "joyful",
    17: "prepared",
    18: "guilty",
    19: "furious",
    20: "nostalgic",
    21: "jealous",
    22: "anticipating",
    23: "embarrassed",
    24: "content",
    25: "devastated",
    26: "sentimental",
    27: "caring",
    28: "trusting",
    29: "ashamed",
    30: "apprehensive",
    31: "faithful",
    32: "questioning",
    33: "acknowledging",
    34: "agreeing",
    35: "consoling",
    36: "encouraging",
    37: "sympathizing",
    38: "wishing",
    39: "suggesting",
    40: "neutral",

}

DIA_EMO_MAP = {
    "surprised": 0,
    "excited": 1,
    "annoyed": 2,
    "proud": 3,
    "angry": 4,
    "sad": 5,
    "grateful": 6,
    "lonely": 7,
    "impressed": 8,
    "afraid": 9,
    "disgusted": 10,
    "confident": 11,
    "terrified": 12,
    "hopeful": 13,
    "anxious": 14,
    "disappointed": 15,
    "joyful": 16,
    "prepared": 17,
    "guilty": 18,
    "furious": 19,
    "nostalgic": 20,
    "jealous": 21,
    "anticipating": 22,
    "embarrassed": 23,
    "content": 24,
    "devastated": 25,
    "sentimental": 26,
    "caring": 27,
    "trusting": 28,
    "ashamed": 29,
    "apprehensive": 30,
    "faithful": 31,
}

TAG_TO_IX = {
    "surprised": 0,
    "excited": 1,
    "annoyed": 2,
    "proud": 3,
    "angry": 4,
    "sad": 5,
    "grateful": 6,
    "lonely": 7,
    "impressed": 8,
    "afraid": 9,
    "disgusted": 10,
    "confident": 11,
    "terrified": 12,
    "hopeful": 13,
    "anxious": 14,
    "disappointed": 15,
    "joyful": 16,
    "prepared": 17,
    "guilty": 18,
    "furious": 19,
    "nostalgic": 20,
    "jealous": 21,
    "anticipating": 22,
    "embarrassed": 23,
    "content": 24,
    "devastated": 25,
    "sentimental": 26,
    "caring": 27,
    "trusting": 28,
    "ashamed": 29,
    "apprehensive": 30,
    "faithful": 31,
    # "neutral": 32
    "questioning": 32,
    "acknowledging": 33,
    "agreeing": 34,
    "consoling": 35,
    "encouraging": 36,
    "sympathizing": 37,
    "wishing": 38,
    "suggesting": 39,
    "neutral": 40,
    # START_TAG: 41,
    # STOP_TAG: 42
}
