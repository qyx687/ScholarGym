import importlib.util
import sys
from pathlib import Path


UTILS_PATH = Path(__file__).resolve().parents[1] / "code" / "utils.py"
CODE_DIR = str(UTILS_PATH.parent)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
SPEC = importlib.util.spec_from_file_location("utils_json_parse_test", UTILS_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_json_from_tag_repairs_qwen_control_chars_and_backslashes():
    response = (
        '<selector_output>{"selected": [], "reasons": {"p": '
        '"uses \\alpha and \\(x\\)"}, "overview": "line one\nline two",}</selector_output>'
    )

    value = MODULE.parse_json_from_tag(response, "selector_output")

    assert value["selected"] == []
    assert value["reasons"]["p"] == r"uses \alpha and \(x\)"
    assert value["overview"] == "line one\nline two"


def test_parse_json_from_tag_does_not_invent_unrecoverable_json():
    assert MODULE.parse_json_from_tag("<selector_output>{not json}</selector_output>", "selector_output") is None
