import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from blockstate import serialize_block


def test_plain_block():
    assert serialize_block("stone") == "stone"


def test_namespaced_passthrough():
    assert serialize_block("minecraft:stone") == "minecraft:stone"


def test_states_no_space_bedrock_syntax():
    out = serialize_block("oak_stairs", {"weirdo_direction": 0, "upside_down_bit": False})
    assert out == 'oak_stairs["upside_down_bit"=false,"weirdo_direction"=0]'  # sorted keys, no space, lowercase bool


def test_string_state_quoted():
    assert serialize_block("stone_block_slab", {"minecraft:vertical_half": "top"}) == 'stone_block_slab["minecraft:vertical_half"="top"]'


def test_empty_states_dict_is_plain():
    assert serialize_block("stone", {}) == "stone"


def test_rejects_injection_in_string_value():
    # A malicious string value must NOT break out of the quotes/brackets.
    with pytest.raises(ValueError):
        serialize_block("command_block", {"k": '"]; setblock ~ ~ ~ tnt; #'})
    with pytest.raises(ValueError):
        serialize_block("stone", {"k": "a]b"})


def test_rejects_injection_in_block_name():
    with pytest.raises(ValueError):
        serialize_block("stone; say hi")
    with pytest.raises(ValueError):
        serialize_block("stone[oops]")


# --- additional coverage ---

def test_int_state_stays_bare():
    assert serialize_block("foo", {"foo": 3}) == 'foo["foo"=3]'


def test_true_lowercased():
    assert serialize_block("foo", {"bar": True}) == 'foo["bar"=true]'


def test_namespaced_state_key_keeps_colon():
    assert serialize_block("foo", {"minecraft:open_bit": True}) == 'foo["minecraft:open_bit"=true]'


def test_uppercase_block_name_rejected():
    with pytest.raises(ValueError):
        serialize_block("Stone")


def test_none_states_is_plain():
    assert serialize_block("stone", None) == "stone"


def test_invalid_state_key_rejected():
    with pytest.raises(ValueError):
        serialize_block("stone", {"bad key": 1})
    with pytest.raises(ValueError):
        serialize_block("stone", {"Bad": 1})
