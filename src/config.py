import torch
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_HERE) == "src":
    _PROJECT_ROOT = os.path.dirname(_HERE)
else:
    _PROJECT_ROOT = _HERE

_PARENT_ROOT = os.path.dirname(_PROJECT_ROOT)


def _resolve_dir(name):
    for base in (_PROJECT_ROOT, _PARENT_ROOT):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            return path
    return os.path.join(_PROJECT_ROOT, name)


DATA_DIR = _resolve_dir("data")
RICO_JSON_DIR = os.path.join(DATA_DIR, "rico_jsons")
RICO_IMAGE_DIR = os.path.join(DATA_DIR, "rico_images")

COMPONENT_CLASSES = [
    "Panel",
    "Button",
    "TextInput",
    "Label",
    "Icon",
    "CheckBox",
    "Dropdown",
    "Slider",
    "Toggle",
    "Image",
]
NUM_CLASSES = len(COMPONENT_CLASSES)

NUM_COORD_BINS = 32

NUM_STYLES = 8

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
COORD_OFFSET = 3
CLASS_OFFSET = COORD_OFFSET + NUM_COORD_BINS
STYLE_OFFSET = CLASS_OFFSET + NUM_CLASSES
VOCAB_SIZE = STYLE_OFFSET + NUM_STYLES

TOKENS_PER_ELEMENT = 6

MAX_ELEMENTS = 25
MAX_SEQ_LEN = MAX_ELEMENTS * TOKENS_PER_ELEMENT + 2

MAX_PANELS = 5
MAX_COMPONENTS = 20
FEATURE_DIM = 5
PANEL_FEATURE_DIM = 5
COMPONENT_FEATURE_DIM = 6
MOCK_DATASET_SIZE = 2000


def normalize_class_id(cls_id):
    return (cls_id + 1) / (NUM_CLASSES + 1)


def denormalize_class_id(cls_norm):
    cls_idx = int(round(cls_norm * (NUM_CLASSES + 1))) - 1
    return max(-1, min(cls_idx, NUM_CLASSES - 1))


def normalize_parent_idx(parent_idx):
    return (parent_idx + 1) / (MAX_PANELS + 1)


def denormalize_parent_idx(parent_norm):
    return int(round(parent_norm * (MAX_PANELS + 1))) - 1


def coord_to_token(val):
    if val < 0.0:
        val = 0.0
    if val > 1.0:
        val = 1.0
    bin_idx = int(val * (NUM_COORD_BINS - 1))
    return bin_idx + COORD_OFFSET


def token_to_coord(tok):
    bin_idx = tok - COORD_OFFSET
    return bin_idx / (NUM_COORD_BINS - 1)


def class_to_token(cls_id):
    return cls_id + CLASS_OFFSET


def token_to_class(tok):
    return tok - CLASS_OFFSET


def style_to_token(style_id):
    return style_id + STYLE_OFFSET


def token_to_style(tok):
    return tok - STYLE_OFFSET


HIDDEN_DIM = 256
NUM_LAYERS = 6
NHEAD = 8
DROPOUT = 0.1

BATCH_SIZE = 128
NUM_EPOCHS = 200
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SAVE_EVERY = 10
EARLY_STOP_PATIENCE = 30

LR_WARMUP_STEPS = 500
LR_MIN = 1e-6

NUM_SAMPLES = 100
TEMPERATURE = 0.9
TOP_K = 0
TOP_P = 0.9

CLASS_COLORS_RGB = {
    0: (230, 230, 230),
    1: (33, 150, 243),
    2: (255, 193, 7),
    3: (60, 60, 60),
    4: (156, 39, 176),
    5: (76, 175, 80),
    6: (255, 87, 34),
    7: (0, 188, 212),
    8: (233, 30, 99),
    9: (139, 195, 74),
}

CLASS_COLORS_VIZ = {
    0: (0.2, 0.4, 0.8, 0.3),
    1: (0.9, 0.3, 0.3, 0.7),
    2: (0.3, 0.8, 0.3, 0.7),
    3: (0.9, 0.9, 0.2, 0.7),
    4: (0.6, 0.3, 0.9, 0.7),
    5: (0.3, 0.9, 0.9, 0.7),
    6: (0.9, 0.6, 0.2, 0.7),
    7: (0.5, 0.5, 0.5, 0.7),
    8: (0.2, 0.7, 0.5, 0.7),
    9: (0.8, 0.4, 0.6, 0.7),
}

STYLE_PALETTES = {
    0: {
        0: ((255, 255, 255), (220, 220, 220), "White"),
        1: ((240, 240, 245), (200, 200, 210), "Light Gray"),
        2: ((227, 242, 253), (144, 202, 249), "Light Blue"),
        3: ((232, 245, 233), (165, 214, 167), "Light Green"),
        4: ((40, 40, 50),    (70, 70, 85),    "Dark"),
        5: ((255, 248, 225), (255, 224, 130), "Cream"),
        6: ((243, 229, 245), (206, 147, 216), "Lavender"),
        7: ((224, 247, 250), (128, 222, 234), "Ice Blue"),
    },
    1: {
        0: ((33, 150, 243),  (255, 255, 255), "Submit"),
        1: ((76, 175, 80),   (255, 255, 255), "OK"),
        2: ((255, 152, 0),   (255, 255, 255), "Warning"),
        3: ((244, 67, 54),   (255, 255, 255), "Delete"),
        4: ((55, 55, 65),    (255, 255, 255), "Dark"),
        5: ((255, 255, 255), (33, 150, 243),  "Outline"),
        6: ((156, 39, 176),  (255, 255, 255), "Action"),
        7: ((0, 150, 136),   (255, 255, 255), "Confirm"),
    },
    2: {
        0: ((255, 255, 255), (180, 180, 180), "Type here..."),
        1: ((255, 255, 255), (180, 180, 180), "Email"),
        2: ((255, 255, 255), (180, 180, 180), "Search..."),
        3: ((255, 255, 255), (180, 180, 180), "Password"),
        4: ((245, 245, 245), (160, 160, 160), "Username"),
        5: ((255, 255, 255), (180, 180, 180), "Phone"),
        6: ((255, 255, 255), (180, 180, 180), "Address"),
        7: ((248, 248, 255), (150, 150, 180), "Message..."),
    },
    3: {
        0: ((245, 245, 245), (60, 60, 60),    "Dark Text"),
        1: ((245, 245, 245), (100, 100, 100),  "Gray Text"),
        2: ((245, 245, 245), (33, 150, 243),   "Blue Text"),
        3: ((245, 245, 245), (76, 175, 80),    "Green Text"),
        4: ((245, 245, 245), (244, 67, 54),    "Red Text"),
        5: ((245, 245, 245), (156, 39, 176),   "Purple Text"),
        6: ((245, 245, 245), (130, 130, 130),  "Light Text"),
        7: ((245, 245, 245), (0, 150, 136),    "Teal Text"),
    },
    4: {
        0: ((156, 39, 176),  (200, 170, 210), "Purple"),
        1: ((33, 150, 243),  (144, 202, 249), "Blue"),
        2: ((244, 67, 54),   (255, 171, 145), "Red"),
        3: ((76, 175, 80),   (165, 214, 167), "Green"),
        4: ((255, 152, 0),   (255, 204, 128), "Orange"),
        5: ((55, 55, 65),    (140, 140, 155), "Dark"),
        6: ((0, 150, 136),   (128, 222, 234), "Teal"),
        7: ((233, 30, 99),   (248, 187, 208), "Pink"),
    },
    5: {
        0: ((76, 175, 80),   (56, 142, 60),   "Green Checked"),
        1: ((33, 150, 243),  (25, 118, 210),  "Blue Checked"),
        2: ((255, 255, 255), (180, 180, 180),  "Unchecked"),
        3: ((156, 39, 176),  (123, 31, 162),  "Purple Checked"),
        4: ((244, 67, 54),   (211, 47, 47),   "Red Checked"),
        5: ((255, 255, 255), (200, 200, 200),  "Unchecked Light"),
        6: ((0, 150, 136),   (0, 121, 107),   "Teal Checked"),
        7: ((255, 152, 0),   (245, 124, 0),   "Orange Checked"),
    },
    6: {
        0: ((255, 255, 255), (120, 120, 120), "Select"),
        1: ((255, 255, 255), (120, 120, 120), "Category"),
        2: ((255, 255, 255), (120, 120, 120), "Country"),
        3: ((255, 255, 255), (120, 120, 120), "Language"),
        4: ((245, 245, 245), (100, 100, 100), "Sort by"),
        5: ((255, 255, 255), (120, 120, 120), "Filter"),
        6: ((255, 255, 255), (120, 120, 120), "Size"),
        7: ((248, 248, 255), (100, 100, 130), "Options"),
    },
    7: {
        0: ((0, 188, 212),   (200, 200, 200), "Teal"),
        1: ((33, 150, 243),  (200, 200, 200), "Blue"),
        2: ((76, 175, 80),   (200, 200, 200), "Green"),
        3: ((255, 152, 0),   (200, 200, 200), "Orange"),
        4: ((244, 67, 54),   (200, 200, 200), "Red"),
        5: ((156, 39, 176),  (200, 200, 200), "Purple"),
        6: ((233, 30, 99),   (200, 200, 200), "Pink"),
        7: ((55, 55, 65),    (180, 180, 180), "Dark"),
    },
    8: {
        0: ((76, 175, 80),   (255, 255, 255), "Green ON"),
        1: ((33, 150, 243),  (255, 255, 255), "Blue ON"),
        2: ((200, 200, 200), (255, 255, 255), "Gray OFF"),
        3: ((156, 39, 176),  (255, 255, 255), "Purple ON"),
        4: ((255, 152, 0),   (255, 255, 255), "Orange ON"),
        5: ((180, 180, 180), (255, 255, 255), "Light OFF"),
        6: ((0, 150, 136),   (255, 255, 255), "Teal ON"),
        7: ((244, 67, 54),   (255, 255, 255), "Red ON"),
    },
    9: {
        0: ((220, 225, 230), (180, 190, 200), "Landscape"),
        1: ((230, 220, 220), (200, 180, 180), "Portrait"),
        2: ((220, 230, 220), (180, 200, 180), "Nature"),
        3: ((225, 225, 235), (190, 190, 210), "Abstract"),
        4: ((235, 230, 220), (210, 200, 180), "Product"),
        5: ((215, 225, 235), (170, 190, 210), "Urban"),
        6: ((230, 225, 215), (200, 190, 170), "Food"),
        7: ((220, 220, 230), (185, 185, 200), "Camera"),
    },
}

IMAGE_SIZE = 64
IMAGE_CHANNELS = 3

OUTPUT_DIR = _resolve_dir("outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")

for d in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR, SAMPLES_DIR]:
    os.makedirs(d, exist_ok=True)
