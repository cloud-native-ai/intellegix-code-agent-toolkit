import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from knowledge import load_blocks, Y_MIN, Y_MAX, FILL_CAP, is_known_block


def test_limits_constants():
    assert Y_MIN == -64 and Y_MAX == 319 and FILL_CAP == 32768


def test_blocks_loads_and_has_common_blocks():
    b = load_blocks()
    for name in ["stone", "gold_block", "oak_planks", "glass", "glowstone"]:
        assert name in b, name
    # gotcha encoded: powered rail canonical bedrock id
    assert b["golden_rail"]["id"] == "minecraft:golden_rail"


def test_is_known_block():
    assert is_known_block("stone")
    assert is_known_block("minecraft:stone")   # namespace tolerated
    assert not is_known_block("definitely_not_a_block_xyz")


def test_every_entry_has_namespaced_id():
    b = load_blocks()
    for name, entry in b.items():
        assert isinstance(entry, dict), name
        assert "id" in entry, name
        assert entry["id"].startswith("minecraft:"), name


def test_at_least_20_blocks():
    b = load_blocks()
    assert len(b) >= 20, len(b)


def test_oak_stairs_documents_states():
    b = load_blocks()
    assert "oak_stairs" in b
    states = b["oak_stairs"].get("states", {})
    assert "weirdo_direction" in states
    assert "upside_down_bit" in states
