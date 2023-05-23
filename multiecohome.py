import streamlit as st
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import pandas as pd

ballons = st.sidebar.checkbox("balloons?", True)

st.markdown('# Generalized Lotka-Volterra sandbox')

st.markdown('''
The Generalized Lotka-Volterra equations define the relation of a group of species as follows (more info below)

$$x' = x\cdot(r + Ax)$$
''')

# doesn't work
# st.sidebar.markdown(f'''
#     <style>
#         section[data-testid="stSidebar"] .css-ng1t4o {{width: 60rem;}}
#         section[data-testid="stSidebar"] .css-1cypcdb {{width: 60rem;}}
#     </style>
# ''',unsafe_allow_html=True)

default_num_species = 10

nice_settings = [
'http://localhost:8501/?num=6&r=-0.4&r=-0.2&r=0.6&r=-0.3&r=-0.1&r=-0.01&A=-0.2&A=0.2&A=0.0&A=0.0&A=0.4&A=0.0&A=-0.3&A=-0.2&A=0.8&A=-0.4&A=0.0&A=0.0&A=0.0&A=0.0&A=-0.2&A=-0.04&A=-0.4&A=0.0&A=0.0&A=0.1&A=0.02&A=-0.2&A=0.5&A=0.0&A=-0.01&A=0.0&A=0.2&A=-0.2&A=-0.1&A=0.05&A=0.0&A=0.0&A=0.5&A=0.0&A=-0.33&A=-0.1&x0=1.0&x0=3.0&x0=3.0&x0=1.0&x0=0.1&x0=0.1',
'http://localhost:8501/?num=10&m=wolf&m=0.1&m=-0.3&m=rabbit&m=0.1&m=-0.08&m=carrot&m=0.1&m=1.27&m=fox&m=0.1&m=-0.1&m=bird&m=0.1&m=-0.1&m=bee&m=0.1&m=-0.05&m=worm&m=0.1&m=-0.1&m=fish&m=0.1&m=-0.1&m=tree&m=0.1&m=2.1&m=berry&m=0.1&m=0.1&A=-0.1&A=0.3&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=-0.4&A=-0.1&A=0.3&A=-0.1&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=-0.3&A=-0.2&A=-0.04&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.2&A=0.0&A=-0.1&A=0.1&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.1&A=-0.1&A=-0.1&A=0.0&A=0.0&A=0.0&A=0.02&A=0.0&A=0.0&A=0.0&A=0.1&A=0.0&A=-0.5&A=-0.1&A=0.0&A=0.0&A=0.02&A=0.02&A=0.0&A=0.0&A=0.0&A=0.0&A=-0.5&A=-0.05&A=-0.1&A=-0.05&A=0.3&A=0.2&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.0&A=0.1&A=-0.2&A=0.0&A=0.02&A=0.0&A=-0.2&A=0.0&A=0.0&A=-0.3&A=0.15&A=-0.4&A=0.0&A=-0.4&A=-0.1&A=0.0&A=0.0&A=0.0&A=0.0&A=0.1&A=0.0&A=-0.01&A=0.0&A=0.0&A=-0.04',
]

# setup

default_species_meta = [
    ['🐺',   'wolf', 0.1, -0.3],
    ['🐇', 'rabbit', 0.1,-0.08],
    ['🥕', 'carrot', 0.1, 1.27],
    ['🦊',    'fox', 0.1, -0.1],
    ['🐦',   'bird', 0.1, -0.1],
    ['🐝',    'bee', 0.1,-0.05],
    ['🐛',   'worm', 0.1, -0.1],
    ['🐟',   'fish', 0.1, -0.1],
    ['🌱',   'tree', 0.1,  2.1],
    ['🍒',  'berry', 0.1,  0.1],
    ['🌿',  'grass', 0.1,  0.1]
]

default_species_meta_columns = ['symbol', 'name', 'initial_density', 'reproduction']

symbols = [species[0] for species in default_species_meta]

st.sidebar.markdown('reproduction is positive for species that survive alone (e.g. plants) and negative otherwise (e.g. carnivores and herbivores).')

st.sidebar.markdown('You can remove a species by selecting its row and pressing Delete')

cross_matrix_columns = [
     'symbol',  '🐺', '🐇',  '🥕',  '🦊', '🐦',  '🐝', '🐛', '🐟',  '🌱',  '🍒',  '🌿'
]
default_cross_matrix = [
        ['🐺',  -0.1,   0.3,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],
        ['🐇',  -0.4,  -0.1,   0.3,  -0.1,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],
        ['🥕',   0.0,  -0.3,  -0.2, -0.04,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],
        ['🦊',   0.0,   0.2,   0.0,  -0.1,   0.1,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],
        ['🐦',   0.0,   0.0,   0.1,  -0.1,  -0.1,   0.0,   0.0,   0.0,  0.02,   0.0,   0.0],
        ['🐝',   0.0,   0.0,   0.1,   0.0,  -0.5,  -0.1,   0.0,   0.0,  0.02,  0.02,  0.01],
        ['🐛',   0.0,   0.0,   0.0,   0.0,   -0.5, -0.05,  -0.1, -0.05,   0.3,   0.2,  0.10],
        ['🐟',   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.1,  -0.2,   0.0,  0.02,   0.0],
        ['🌱',   0.0,  -0.2,   0.0,   0.0,  -0.3,  0.15,  -0.4,   0.0,  -0.4,  -0.1, -0.05],
        ['🍒',   0.0,   0.0,   0.0,   0.0,   0.1,   0.0, -0.01,   0.0,   0.0, -0.04,   0.0],
        ['🌿',   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0, -0.04],
]

if 'initialized' not in st.session_state:
    print('Fetching data from query_params or default')
    st.session_state['initialized'] = True

    get_query_params = st.experimental_get_query_params()

    default_num_species = int(get_query_params.get('num', [default_num_species])[0])
    if 'm' in get_query_params:
        default_species_meta = np.array(get_query_params['m'])
        default_species_meta = default_species_meta.reshape(default_num_species,4)
        default_species_meta[:,2:3] = default_species_meta[:,2:3].astype(float)

    if 'A' in get_query_params:
        qp_cross_matrix = [float(a) for a in get_query_params['A']]
        symbols = [species[0] for species in default_species_meta]
        default_cross_matrix = [[symbol] + qp_cross_matrix[i*len(symbols):(i+1)*len(symbols)] for i,symbol in enumerate(symbols)]

species_meta = pd.DataFrame(columns=default_species_meta_columns, data=default_species_meta)

num_species = st.sidebar.slider('Number of species', 1, 10, value=default_num_species)

species_meta = species_meta.iloc[0:num_species]
species_meta_list = st.sidebar.experimental_data_editor(species_meta, num_rows="dynamic")

# TODO check if a row was removed and do
# num_species = min(species_meta_list, num_species)
# default_cross_matrix filter symbols
# or maybe cross filter symbols

cross_col_names = ['symbol'] + [row[0] for row in default_cross_matrix]
default_cross_matrix = pd.DataFrame(columns=cross_col_names, data=default_cross_matrix)
default_cross_matrix = default_cross_matrix.set_index('symbol')
default_cross_matrix = default_cross_matrix.iloc[0:num_species,0:num_species]
cross = st.sidebar.experimental_data_editor(default_cross_matrix)

# experimental_data_editor only returns lists, we need to rebuild the df

# species_meta_df = pd.DataFrame(columns=default_species_meta_columns, data=species_meta_list)
# species_meta_df = species_meta_df.set_index('symbol')

# cross_df = pd.DataFrame(columns=cross_matrix_columns, data=default_cross_matrix)
# cross_df = cross_df.set_index('symbol')


# TODO getting from params it's gonna be a bit tricky now

# if len(species_meta_df) < num_species:
#     num_species = len(species_meta_df)
#     current_species = species_meta_df.index.to_list()

#     previous_species = species_data.index.to_list()
#     removed_species = [species for species in previous_species if species not in current_species]
#     species_data = species_data.drop(labels=removed_species, axis=0)
#     species_data = species_data.drop(labels=removed_species, axis=1)

experiment_params = {"num": len(species_meta_list), "m": species_meta_list.to_numpy().flatten(), "A": cross.to_numpy().flatten()}
st.experimental_set_query_params(**experiment_params)


st.sidebar.markdown('The interaction matrix is negative for predators and the self-predation and positive for the species that actually nourish or help.')

def glv(x, r, A):

    x[x < 0.00000001] = 0 # prevent numerical problems

    dxdt = x*(r + A.dot(x))

    return dxdt


A = cross.to_numpy()
r = species_meta_list['reproduction'].astype(float).to_numpy()
x0 = species_meta_list['initial_density'].astype(float)
names = species_meta_list['symbol']
equilibrium = -np.linalg.inv(A).dot(r)

st.markdown(f"Given your parameters we have the following results")

progress_bar = st.progress(0)
status_text = st.markdown('The latest population numbers are:')
last_population = st.empty()
st.markdown("Equilibrium point")
st.table(pd.DataFrame(equilibrium.reshape(1,len(equilibrium)), columns=names))

community_matrix = equilibrium*A
eigen = np.linalg.eig(community_matrix)
eigens = eigen[0].tolist()
st.markdown("The Eigenvalues in this current experiment:")
st.json(eigens, expanded=False)

st.markdown('### Density of each species')
numiterations = st.number_input('numiterations', value= 150)

sim = np.array([x0])
sim_df = pd.DataFrame(sim, columns=names)

chart = st.line_chart(sim_df)

for i in range(numiterations):
    # Update progress bar.
    progress_bar.progress((i + 1)/numiterations)

    new_row = sim[-1] + glv(sim[-1], r, A)
    new_row[new_row < 0.0000001] = 0 # prevent numerical problems

    # Update status text.
    last_population.table(pd.DataFrame(new_row.reshape(1,len(new_row)), columns=names))

    sim = np.vstack([sim, new_row])

    df_new_row = pd.DataFrame([new_row], columns=names)

    # Append data to the chart.
    chart.add_rows(df_new_row)

st.markdown('## Share the url at my [discord](https://discord.gg/RDEXTjf) or [tweet me](https://twitter.com/catclicli) if you found something interesting!')
st.markdown("## 📣[share]()")

st.markdown('Reading the query params is not yet supported sorry!')

chart_data = pd.DataFrame(sim, columns=names)
st.dataframe(chart_data)


if ballons:
    st.balloons()
