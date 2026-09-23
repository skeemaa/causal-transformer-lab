import causal_transformer_lab


def test_package_imports() -> None:
    assert causal_transformer_lab.__doc__ is not None, (
        "causal transformer lab missing its docstring"
    )
