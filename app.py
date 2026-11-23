from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st
from demand import benchmark

ROOT = Path(__file__).parent
st.set_page_config(page_title='BikeWindow ML', page_icon='🚲', layout='wide')
st.title('🚲 BikeWindow ML')
st.caption('Local bike-demand estimates with an uncertainty window')
st.warning('Educational simulation: trained on fictional data. Not a real station forecast or inventory recommendation.')

@st.cache_resource
def load_demo():
    return benchmark(pd.read_csv(ROOT / 'data/daily_demand.csv'))

model, metrics, details = load_demo()
scenario, evaluation, learning = st.tabs(['Try a day', 'Held-out evaluation', 'How it works'])
with scenario:
    chosen = st.date_input('Date', date(2026, 1, 5), min_value=date(2024, 1, 1), max_value=date(2026, 12, 31))
    temperature = st.slider('Forecast temperature (°C)', -5, 45, 24)
    rain = st.slider('Forecast rain (mm)', 0, 80, 0)
    holiday = st.checkbox('Public holiday')
    row = pd.DataFrame([dict(date=chosen.isoformat(), temperature_c=temperature, rain_mm=rain, holiday=int(holiday))])
    result = model.predict(row).iloc[0]
    a, b = st.columns(2)
    a.metric('Expected rentals', f'{result.expected:.1f}')
    b.metric('Nominal 90% window', f'{int(result.lower)}–{int(result.upper)}')
    st.info('The range describes daily count uncertainty, not a confidence interval for the mean. Coverage is measured below; 90% is a target, not a promise for this date.')
    if temperature < 8 or temperature > 36 or rain > 30:
        st.warning('This weather is rare or outside the typical demo conditions. Extrapolation can be unreliable.')
    output = row.assign(expected=result.expected, lower=result.lower, upper=result.upper)
    st.download_button('Download this estimate (CSV)', output.to_csv(index=False), 'estimate.csv', 'text/csv')
with evaluation:
    st.write('Chronological 60% training / 20% calibration / 20% test. Test labels do not fit the model or interval.')
    a, b, c = st.columns(3)
    a.metric('Model MAE', f"{metrics['mae']:.2f}")
    b.metric('Weekday baseline MAE', f"{metrics['weekday_baseline_mae']:.2f}")
    c.metric('Test interval coverage', f"{metrics['test_coverage']:.1%}")
    st.line_chart(details.set_index('date')[['actual', 'expected', 'lower', 'upper']])
    st.dataframe(details, hide_index=True)
    st.download_button('Download held-out predictions', details.to_csv(index=False), 'holdout.csv', 'text/csv')
with learning:
    st.markdown('**Calendar + assumed weather → scaled features → Poisson regression → calibrated absolute residual window.**\n\nA second, later block sets the window width. A final untouched block measures both prediction error and coverage.\n\nStandard split-conformal coverage relies on exchangeability; chronological time series and distribution shift do not satisfy that guarantee in general. The constant-width interval also misses changing uncertainty across conditions. See docs/LEARNING.md for exercises.')
