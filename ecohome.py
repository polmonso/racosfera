import streamlit as st
import numpy as np
import pandas as pd
import time
from autograd import grad, jacobian
import plotly.express as px


st.markdown('# Generalized Lotka-Volterra sandbox')

ballons = st.sidebar.checkbox("ballons?", True)

progress_bar = st.progress(0)
status_text = st.empty()
equilibrium_text = st.empty()
equilibrium_eigenvalues_text = st.empty()

species = ['wolf', 'rabbit', 'grass']

num_species = len(species)

nice_settings = [
    {
        'x0': [0.6, 1.44, 10.0],
        'r': [-0.3, 0.08, 0.82],
        'A': [[-0.1, 0.2, 0],
              [-0.3, 0.1, 0.1],
              [0, -0.2, -0.04]]
    },
    'http://localhost:8502/?r=-0.3&r=0.08&r=0.8&A=-0.1&A=0.2&A=0.0&A=-0.3&A=-0.1&A=0.1&A=0.0&A=-0.2&A=-0.04&x0=0.6&x0=1.5&x0=10.0',
    'http://localhost:8502/?r=-0.62&r=0.86&r=-1.0&A=-0.05&A=0.25&A=0.0&A=-0.2&A=-0.1&A=0.0&A=0.0&A=0.0&A=1.0&x0=1.46&x0=2.79&x0=0.0'
]

x0 = [0.6, 1.5, 10.0]
r = [-0.3, 0.08, 0.8]
A = np.array([
    [    -0.1,      0.2,    0],
    [   -0.3,     -0.1,   0.1],
    [     0,     -0.2,    -0.04]
])

if 'initialized' not in st.session_state:
    print('Fetching data from query_params or default')
    st.session_state['initialized'] = True
    get_query_params = st.experimental_get_query_params()

    A = np.array(get_query_params.get('A', A.flatten())).astype(float).reshape(3,3)
    x0 = np.array(get_query_params.get('x0', x0)).astype(float)
    r = np.array(get_query_params.get('r', r)).astype(float)

    st.session_state['A'] = A
    st.session_state['x0'] = x0
    st.session_state['r'] = r

if 'A' in st.session_state:
    A = st.session_state['A']

if 'x0' in st.session_state:
    x0 = st.session_state['x0']

if 'r' in st.session_state:
    r = st.session_state['r']

print('')
print(f'x0: {x0} r: {r} A: {A}')

wolves_0 = st.sidebar.slider('Starting wolves', 0.0, 10.0, value=float(x0[0]))
rabbits_0 = st.sidebar.slider('Starting rabbits', 0.0, 10.0, value=float(x0[1]))
grass_0 = st.sidebar.slider('Starting grass', 0.0, 10.0, value=float(x0[2]))

x0 = [wolves_0, rabbits_0, grass_0]

p = np.array([x0])

r_wolves = st.sidebar.slider('Rate wolves', -1.0, 1.0, float(r[0]))
r_rabbits = st.sidebar.slider('Rate rabbits', -1.0, 1.0, float(r[1]))
r_grass = st.sidebar.slider('Rate grass', -1.0, 1.0, float(r[2]))
r = np.array([r_wolves, r_rabbits, r_grass])

df_p = pd.DataFrame(p, columns=species)

st.sidebar.markdown("Interaction Matrix A")
# interaction matrix

A = st.sidebar.experimental_data_editor(A)

def glv(x, t, r, A):

     x[x < 0.00000001] = 0 # prevent numerical problems

     dxdt = x*(r + A.dot(x))

     return dxdt

equilibrium = -np.linalg.inv(A).dot(r)

st.markdown('''The Jacobian at the equilibrium points is called community matrix.\\
             For LGV that is $M = J|x^* = D(x^*)·A$''')

st.markdown(f"Equilibrium point: {equilibrium}")

community_matrix = equilibrium*A

eigen = np.linalg.eig(community_matrix)

eigens = eigen[0].tolist()

st.markdown("Eigenvalues:")

eigens

st.markdown("If $Re(\lambda_1) < 0 → x^*$ the equilibrium is stable")

print("")

st.markdown('### Density of each species')
numiterations = st.number_input('numiterations', value= 150)

chart = st.line_chart(df_p)


for i in range(numiterations):
    # Update progress bar.
    progress_bar.progress((i + 1)/numiterations)

    foo = A.dot(p[-1])
    # dx = p[-1]*(r + a.dot(p[-1]))
    # xi = dx + p[-1]
    # new_row = np.array([xi])

    new_row = p[-1] + glv(p[-1], 1, r, A)

    new_row[new_row < 0.0000001] = 0 # prevent numerical problems

    # Update status text.
    status_text.text(f'The latest population numbers are: {new_row}')

    p = np.vstack([p, new_row])

    df_new_row = pd.DataFrame([new_row], columns=species)

    # Append data to the chart.
    chart.add_rows(df_new_row)

    # Pretend we're doing some computation that takes time.
    # time.sleep(0.01)

# status_text.text('Done!')

chart_data = pd.DataFrame(p, columns=species)

st.dataframe(chart_data, use_container_width=True)

# chart_data['rabbit+grass'] = chart_data['rabbit'] + chart_data['grass']

# chart_xy = st.line_chart(chart_data, x='wolf', y='rabbit')

fig = px.scatter(
    chart_data,
    x="wolf",
    y="rabbit",
    hover_name=chart_data.index,
    color=chart_data.index
)
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

if ballons:
    st.balloons()

experiment_params = {"r": r.tolist(), "A": A.flatten(), "x0": x0}

st.experimental_set_query_params(**experiment_params)

st.markdown('## Share the url with [me](mailto:pol.monso@somenergia.coop) if you found something interesting!')
experiment_get_params = st.experimental_get_query_params()

st.markdown("## [$\Delta$share]()")