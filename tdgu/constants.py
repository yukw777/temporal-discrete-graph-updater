# One arg state relations
OPEN = "open"
CLOSED = "closed"
CHOPPED = "chopped"
ROASTED = "roasted"
DICED = "diced"
BURNED = "burned"
FRIED = "fried"
GRILLED = "grilled"
CONSUMED = "consumed"
SLICED = "sliced"
CUT = "cut"
COOKED = "cooked"
UNCUT = "uncut"
RAW = "raw"
EDIBLE = "edible"
INEDIBLE = "inedible"
CUTTABLE = "cuttable"
DRINKABLE = "drinkable"
SHARP = "sharp"
COOKABLE = "cookable"
NEEDS_COOKING = "needs_cooking"

# Two arg relations
NORTH_OF = "north_of"
SOUTH_OF = "south_of"
EAST_OF = "east_of"
WEST_OF = "west_of"
PART_OF = "part_of"
AT = "at"
IN = "in"
IS = "is"
ON = "on"
NEEDS = "needs"
BASE = "base"

# Three arg relations
LINK = "link"

# Food states
FOOD_STATES = [
    SLICED,
    DICED,
    CHOPPED,
    CUT,
    UNCUT,
    COOKED,
    BURNED,
    GRILLED,
    FRIED,
    ROASTED,
    RAW,
    EDIBLE,
    INEDIBLE,
]

TWO_ARGS_RELATIONS = [
    IN,
    ON,
    AT,
    WEST_OF,
    EAST_OF,
    NORTH_OF,
    SOUTH_OF,
    PART_OF,
    NEEDS,
]
ONE_ARG_STATE_RELATIONS = [
    CHOPPED,
    ROASTED,
    DICED,
    BURNED,
    OPEN,
    FRIED,
    GRILLED,
    CONSUMED,
    CLOSED,
    SLICED,
    UNCUT,
    RAW,
]
IGNORE_RELATIONS = [
    CUTTABLE,
    EDIBLE,
    DRINKABLE,
    SHARP,
    INEDIBLE,
    CUT,
    COOKED,
    COOKABLE,
    NEEDS_COOKING,
]
PREDICATES_TO_DISCARD = {
    "ingredient_1",
    "ingredient_2",
    "ingredient_3",
    "ingredient_4",
    "ingredient_5",
    "out",
    "free",
    "used",
    "cooking_location",
    "link",
}
CONSTANT_NAMES = {
    "P": "player",
    "I": "player",
    "RECIPE": "cookbook",
}

EVENT_TYPES = [
    "pad",
    "start",
    "end",
    "node-add",
    "node-delete",
    "edge-add",
    "edge-delete",
]
EVENT_TYPE_ID_MAP = {v: k for k, v in enumerate(EVENT_TYPES)}

COMMANDS_TO_IGNORE = ["look", "examine", "inventory"]

FOOD_KINDS = {
    "red tuna": "tuna",
    "white tuna": "tuna",
    "red onion": "onion",
    "white onion": "onion",
    "yellow onion": "onion",
    "red potato": "potato",
    "yellow potato": "potato",
    "purple potato": "potato",
    "red apple": "apple",
    "yellow apple": "apple",
    "green apple": "apple",
    "red hot pepper": "hot pepper",
    "green hot pepper": "hot pepper",
    "red bell pepper": "bell pepper",
    "yellow bell pepper": "bell pepper",
    "green bell pepper": "bell pepper",
    "orange bell pepper": "bell pepper",
}

FOOD_COLORS = {
    "tuna": {"red", "white"},
    "onion": {"red", "white", "yellow"},
    "potato": {"red", "yellow", "purple"},
    "apple": {"red", "yellow", "green"},
    "hot pepper": {"red", "green"},
    "bell pepper": {"red", "yellow", "green", "orange"},
}
