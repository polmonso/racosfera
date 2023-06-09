import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt



st.markdown('# Generalized Lotka-Volterra sandbox')

st.markdown('''
The Generalized Lotka-Volterra equations define the relation of a group of species as follows (more info below)

$$x' = x\cdot(r + Ax)$$
''')



ballons = st.sidebar.checkbox("balloons?", True)


species = ['wolf', 'rabbit', 'carrot']
species = ['🐺', '🐇', '🥕']


num_species = len(species)

nice_settings = [
    'http://localhost:8502/?r=-0.3&r=0.08&r=0.8&A=-0.1&A=0.2&A=0.0&A=-0.3&A=-0.1&A=0.1&A=0.0&A=-0.2&A=-0.04&x0=0.6&x0=1.5&x0=10.0',
    'http://localhost:8502/?r=-0.62&r=0.86&r=-1.0&A=-0.05&A=0.25&A=0.0&A=-0.2&A=-0.1&A=0.0&A=0.0&A=0.0&A=1.0&x0=1.46&x0=2.79&x0=0.0',
    'http://localhost:8503/?r=-0.57&r=0.08&r=0.8&A=-0.1&A=0.2&A=0.0&A=-0.3&A=-0.1&A=0.1&A=0.0&A=-0.2&A=-0.04&x0=0.6&x0=1.5&x0=10.0',
    'http://localhost:8501/?r=-0.45&r=0.08&r=0.8&A=-0.1&A=0.25&A=0.0&A=-0.3&A=-0.1&A=0.1&A=0.0&A=-0.2&A=-0.04&x0=0.6&x0=1.5&x0=10.0',
    'http://localhost:8501/?r=-0.32&r=0.04&r=0.48&A=-0.1&A=0.4&A=0.0&A=-0.3&A=-0.1&A=0.1&A=0.0&A=-0.2&A=-0.04&x0=0.6&x0=1.5&x0=10.0',
    'http://localhost:8501/?r=-0.32&r=-0.05&r=0.48&A=-0.1&A=0.4&A=0.0&A=-0.3&A=-0.1&A=0.1&A=0.0&A=-0.2&A=-0.04&x0=0.6&x0=1.5&x0=10.0'
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


wolves_0 = st.sidebar.slider('Starting wolves', 0.0, 10.0, value=float(x0[0]))
rabbits_0 = st.sidebar.slider('Starting rabbits', 0.0, 10.0, value=float(x0[1]))
carrot_0 = st.sidebar.slider('Starting carrot', 0.0, 10.0, value=float(x0[2]))

x0 = [wolves_0, rabbits_0, carrot_0]

p = np.array([x0])

r_wolves = st.sidebar.slider('Rate wolves', -1.0, 1.0, float(r[0]))
r_rabbits = st.sidebar.slider('Rate rabbits', -1.0, 1.0, float(r[1]))
r_carrot = st.sidebar.slider('Rate carrot', -1.0, 1.0, float(r[2]))
r = np.array([r_wolves, r_rabbits, r_carrot])

df_p = pd.DataFrame(p, columns=species)

st.sidebar.markdown('''Interaction Matrix A 🐺 🐇 🥕''')
# interaction matrix

A = st.sidebar.experimental_data_editor(A)

print(f'sidebar\nx0: {x0} r: {r} A: {A}')

def glv(x, t, r, A):

    x[x < 0.00000001] = 0 # prevent numerical problems

    dxdt = x*(r + A.dot(x))

    return dxdt

def glv_mesh(mesh_x, r_, A):

    mesh_x[mesh_x < 0.00000001] = 0 # prevent numerical problems

    r_mesh = np.tile(r_, (mesh_x.shape[1], 1))

    mesh_dxdt = mesh_x.T*(r_mesh + A.dot(mesh_x).T)

    mesh_dxdt[mesh_dxdt < 0.00000001] = 0

    return mesh_dxdt

equilibrium = -np.linalg.inv(A).dot(r)

st.markdown(f"Given your parameters we have the following results")

progress_bar = st.progress(0)
status_text = st.empty()
equilibrium_text = st.empty()
equilibrium_eigenvalues_text = st.empty()

equilibrium_text.text(f"Equilibrium point: {equilibrium}")

community_matrix = equilibrium*A

eigen = np.linalg.eig(community_matrix)

eigens = eigen[0].tolist()


st.markdown('### Density of each species')
numiterations = st.number_input('numiterations', value= 150)

chart = st.line_chart(df_p)


for i in range(numiterations):
    # Update progress bar.
    progress_bar.progress((i + 1)/numiterations)

    new_row = p[-1] + glv(p[-1], 1, r, A)

    new_row[new_row < 0.0000001] = 0 # prevent numerical problems

    # Update status text.
    status_text.text(f'The latest population numbers are: {new_row}')

    p = np.vstack([p, new_row])

    df_new_row = pd.DataFrame([new_row], columns=species)

    # Append data to the chart.
    chart.add_rows(df_new_row)



chart_data = pd.DataFrame(p, columns=species)

st.dataframe(chart_data, use_container_width=True)

# chart_data['rabbit+carrot'] = chart_data['rabbit'] + chart_data['carrot']

# chart_xy = st.line_chart(chart_data, x='wolf', y='rabbit')

fig = px.scatter(
    chart_data,
    x="🐺",
    y="🐇",
    hover_name=chart_data.index,
    color=chart_data.index,
    labels={'🐺':'🐺 wolves','🐇':'🐇 rabbits'}
)
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

fig2 = px.scatter_3d(chart_data,
    x="🐺",
    y="🐇",
    z="🥕",
    hover_name=chart_data.index,
    color=chart_data.index)

st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

experiment_params = {"r": r.tolist(), "A": A.flatten(), "x0": x0}

st.experimental_set_query_params(**experiment_params)

st.markdown('## 📣 Share the url at my [discord](https://discord.gg/RDEXTjf) or [tweet me](https://twitter.com/catclicli) if you found something interesting!')
experiment_get_params = st.experimental_get_query_params()

st.markdown('''\
## Further explanation

The Generalized Lotka-Volterra equations define the relation of a group of species via three parameters:

$r$: The rate of reproduction in the absence of other species

$A$: A matrix that crosses each species against each other and itself. It represents the impact of each species towards the others.

$x0$: The initial density of each species. This is in fact somewhat irrellevant in the long term,\
it only determines in which local minima the system will orbit around, if there where more than one

Put togehter we get $x'$ the variation of density as

$ x' = x\cdot(r + Ax)$
''')

st.markdown('''\
### Important take aways

So in the A matrix, the diagonal shows how does one species impact itself.
It models the carrying capacity effect, that is, in the absence of other species, how much does my own species hinder my growth?

That means that for predators or species that **need another species to survive** it should be negative.

The difference between the rate r is that r models the rate of reproduction if unbounded. That is, if no other species **nor my own** impact me.

Again, for species that need other species to survive, that would be negative, otherwise there will always be a certain density that, in the absence of any other species, mine can find a positive equilibrium.
''')

st.markdown('''
### About equilibrium points

The Jacobian at the equilibrium points is called community matrix.

For LGV that is $M = J|x^* = D(x^*)·A$

The Jacobian tells us what will happen if we move slightly away from the evaluated point (noted with $|x^*$).

Therefore if the real part of the eigenvalues is negative, then the equilibrium is stable because the system will return to the equilibria

$Re(\lambda_1) < 0 → x^*$
''')
st.markdown("The Eigenvalues in this current experiment:")

eigens

st.markdown('''
## Bibliography

I was inspired by many literature, but the best source of documentation I found for this simulator
are [the talks of Stefano Allesina](https://stefanoallesina.github.io/Sao_Paulo_School/)
which you can also find [online](https://www.youtube.com/watch?v=loU0O4-dHkw)
''')


def streamplot():

    # plotly streamlines do not look good either, there's probably something wrong in my reshaping

    wolves, rabbits, carrot = np.meshgrid(np.arange(0.1,3.1,0.1), np.arange(0.1, 3.1, 0.1), np.arange(5, 8, 0.1))
    mesh = np.vstack([wolves.ravel(), rabbits.ravel(), carrot.ravel()])

    meshdot = glv_mesh(mesh, r, A)

    # ff streamline can't handle dx/dt = 0
    meshdot[meshdot < 0.0001] = 0.0001

    u = meshdot.T[0][:].reshape(30,30,30)
    v = meshdot.T[1][:].reshape(30,30,30)
    w = meshdot.T[2][:].reshape(30,30,30)

    nslice = st.slider('slice wolves vs rabbits', 0, 30, 14)

    fig = ff.create_streamline(np.arange(0.1,3.1,0.1), np.arange(0.1,3.1,0.1), u[:,:,nslice], v[:,:,nslice], arrow_scale=.1)

    st.plotly_chart(fig)

    fig = ff.create_streamline(np.arange(0.1,3.1,0.1), np.arange(5, 8, 0.1), u[nslice,:,], v[nslice,:,:], arrow_scale=.1)
    st.plotly_chart(fig)



    # There's something wrong on how we go from a exploded horizontal array to a streamplot who expects a chessboard overlap
    # maybe it's because we're going from a 3d streamplot to a 2d slice, but who knows

    # plt.style.use('dark_background')
    # fig, ax = plt.subplots()
    # wolves2d = wolves[:,:,0]
    # rabbits2d = rabbits[:,:,0]
    # nslice = st.slider('slice wolves vs rabbits', 0, 30, 14)
    # u = meshdot.T[1][(nslice*900):((nslice+1)*900)].reshape(30,30)
    # v = meshdot.T[2][(nslice*900):((nslice+1)*900)].reshape(30,30)
    # ax.streamplot(wolves2d, rabbits2d, u, v)

    # st.pyplot(fig)

    # doesnt work, non-interactive gui it says
    # u = meshdot.T[1][:].reshape(30,30,30)
    # v = meshdot.T[2][:].reshape(30,30,30)
    # w = meshdot.T[2][:].reshape(30,30,30)

    # ax2 = plt.figure().add_subplot(projection='3d')
    # ax2.quiver(wolves, rabbits, carrot, u, v, w, length=0.1, normalize=True)

    # plt.show()

# streamplot()

if ballons:
    st.balloons()
