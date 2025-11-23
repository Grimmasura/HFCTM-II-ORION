from mih_iie.layers.l2_majorana_array import (
    BraidWord,
    braid_word_from_generators,
    build_e8_coxeter_matrix,
    compile_reflection_sequence,
    simple_reflection_basis,
)


def test_coxeter_matrix_values_are_expected():
    cox = build_e8_coxeter_matrix()
    assert set(cox.values()).issubset({2, 3})
    assert len(cox) == 8 * 7  # off-diagonal entries


def test_braid_relations_hold_for_connected_pair():
    cox = build_e8_coxeter_matrix()
    connected = [(i, j) for (i, j), m in cox.items() if i < j and m == 3]
    assert connected, "E8 Coxeter graph should have connected pairs"
    i, j = connected[0]

    lhs = BraidWord((i, j, i)).normalize(cox)
    rhs = BraidWord((j, i, j)).normalize(cox)
    assert lhs.generators == rhs.generators


def test_commuting_generators_are_sorted_in_normal_form():
    cox = build_e8_coxeter_matrix()
    commuting = [(i, j) for (i, j), m in cox.items() if i < j and m == 2]
    assert commuting, "There should be commuting simple reflections"
    i, j = max(commuting)  # pick a larger pair to exercise reordering

    word = BraidWord((j, i)).normalize(cox)
    assert word.generators == (i, j)


def test_compile_reflection_sequence_normalizes_input():
    cox = build_e8_coxeter_matrix()
    word = compile_reflection_sequence([2, 1, 2], coxeter=cox)
    # If s1 and s2 commute in this numbering, this should sort; otherwise braid relation triggers
    normalized = braid_word_from_generators(word.generators).normalize(cox)
    assert word.generators == normalized.generators


def test_simple_reflection_basis_provides_all_generators():
    basis = simple_reflection_basis()
    assert len(basis) == 8
    assert {w.generators for w in basis} == {(i,) for i in range(1, 9)}
