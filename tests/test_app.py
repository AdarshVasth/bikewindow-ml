from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_demo_controls_and_evaluation():
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run(timeout=30)
    assert not app.exception
    initial = app.metric[0].value
    app.slider[1].set_value(25).run()
    assert not app.exception
    assert app.metric[0].value != initial
    app.checkbox[0].check().run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert 'Model MAE' in [item.label for item in app.metric]
