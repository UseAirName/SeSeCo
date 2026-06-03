import yaml


class Config:
    def __init__(self, path):
        with open(path) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
            self.yaml_config = cfg

            cfg_misc = cfg["misc"]
            self.exp_name  = cfg_misc["name"]
            self.data_dir  = cfg_misc["data_dir"]
            self.cache_dir = cfg_misc["cache_dir"]
            self.save_dir  = cfg_misc["save_dir"]
            self.verbose   = bool(cfg_misc["verbose"])
            self.qp        = int(cfg_misc["quality_parameter"])

            cfg_encoder = cfg["encoder"]
            self.image_dim         = int(cfg_encoder["image_dim"])
            self.seg_dim           = int(cfg_encoder["seg_dim"])
            self.coarsening_factor = int(cfg_encoder["coarsening_factor"])
            self.q_step            = float(cfg_encoder["quantize_step"])
            self.threshold         = float(cfg_encoder["threshold"])

            cfg_decoder = cfg["decoder"]
            self.conditional_scale = float(cfg_decoder["conditional_scale"])
            self.control_scale     = float(cfg_decoder["control_scale"])
            self.guidance_scale    = float(cfg_decoder["guidance_scale"])
            self.repeat            = int(cfg_decoder["repeat"])
            self.steps             = int(cfg_decoder["steps"])
            self.prompt            = cfg_decoder["prompt"]
            self.negative_prompt   = cfg_decoder["negative_prompt"]

            self.exp_name += "_QP" + str(self.qp)


class Preferences:
    def __init__(self, path):
        with open(path) as file:
            parser = yaml.load(file, Loader=yaml.FullLoader)
            self.list_val = {}
            preference_list = parser["label_importance"]
            for i, v in enumerate(preference_list):
                self.list_val[i] = v


LABELS = {
    0: "wall",
    1: "building",
    2: "sky",
    3: "floor",
    4: "tree",
    5: "ceiling",
    6: "road",
    7: "bed",
    8: "windowpane",
    9: "grass",
    10: "cabinet",
    11: "sidewalk",
    12: "person",
    13: "earth",
    14: "door",
    15: "table",
    16: "mountain",
    17: "plant",
    18: "curtain",
    19: "chair",
    20: "car",
    21: "water",
    22: "painting",
    23: "sofa",
    24: "shelf",
    25: "house",
    26: "sea",
    27: "mirror",
    28: "rug",
    29: "field",
    30: "armchair",
    31: "seat",
    32: "fence",
    33: "desk",
    34: "rock",
    35: "wardrobe",
    36: "lamp",
    37: "bathtub",
    38: "railing",
    39: "cushion",
    40: "base",
    41: "box",
    42: "column",
    43: "signboard",
    44: "chest of drawers",
    45: "counter",
    46: "sand",
    47: "sink",
    48: "skyscraper",
    49: "fireplace",
    50: "refrigerator",
    51: "grandstand",
    52: "path",
    53: "stairs",
    54: "runway",
    55: "case",
    56: "pool table",
    57: "pillow",
    58: "screen door",
    59: "stairway",
    60: "river",
    61: "bridge",
    62: "bookcase",
    63: "blind",
    64: "coffee table",
    65: "toilet",
    66: "flower",
    67: "book",
    68: "hill",
    69: "bench",
    70: "countertop",
    71: "stove",
    72: "palm",
    73: "kitchen island",
    74: "computer",
    75: "swivel chair",
    76: "boat",
    77: "bar",
    78: "arcade machine",
    79: "hovel",
    80: "bus",
    81: "towel",
    82: "light",
    83: "truck",
    84: "tower",
    85: "chandelier",
    86: "awning",
    87: "streetlight",
    88: "booth",
    89: "television receiver",
    90: "airplane",
    91: "dirt track",
    92: "apparel",
    93: "pole",
    94: "land",
    95: "bannister",
    96: "escalator",
    97: "ottoman",
    98: "bottle",
    99: "buffet",
    100: "poster",
    101: "stage",
    102: "van",
    103: "ship",
    104: "fountain",
    105: "conveyer belt",
    106: "canopy",
    107: "washer",
    108: "plaything",
    109: "swimming pool",
    110: "stool",
    111: "barrel",
    112: "basket",
    113: "waterfall",
    114: "tent",
    115: "bag",
    116: "minibike",
    117: "cradle",
    118: "oven",
    119: "ball",
    120: "food",
    121: "step",
    122: "tank",
    123: "trade name",
    124: "microwave",
    125: "pot",
    126: "animal",
    127: "bicycle",
    128: "lake",
    129: "dishwasher",
    130: "screen",
    131: "blanket",
    132: "sculpture",
    133: "hood",
    134: "sconce",
    135: "vase",
    136: "traffic light",
    137: "tray",
    138: "ashcan",
    139: "fan",
    140: "pier",
    141: "crt screen",
    142: "plate",
    143: "monitor",
    144: "bulletin board",
    145: "shower",
    146: "radiator",
    147: "glass",
    148: "clock",
    149: "flag"
}