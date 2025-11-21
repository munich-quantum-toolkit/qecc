"""Testing of layouts. Just runs an instance each."""

from mqt.qecc.cococo import layouts


def test_scalable_layout():
    """Just runs an instance of every layout type. If no error occurs all is fine. TODO more sophisticated tests."""
    m = 5
    n = 5
    factories = []
    remove_edges = False
    try:
        _, _, _ = layouts.gen_layout_scalable("single", m, n, factories, remove_edges)
    except:  # noqa: E722
        raise ValueError("Problem with single layout")
    try:
        _, _, _ = layouts.gen_layout_scalable("pair", m, n, factories, remove_edges)
    except:  # noqa: E722
        raise ValueError("Problem with pair layout")
    try:
        _, _, _ = layouts.gen_layout_scalable("triple", m, n, factories, remove_edges)
    except:  # noqa: E722
        raise ValueError("Problem with triple layout")
    try:
        _, _, _ = layouts.gen_layout_scalable("hex", m, n, factories, remove_edges)
    except:  # noqa: E722
        raise ValueError("Problem with hex layout")
