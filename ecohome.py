import streamlit as st
import numpy as np
import pandas as pd
import time
from autograd import grad, jacobian

progress_bar = st.progress(0)
status_text = st.empty()
equilibrium_text = st.empty()
equilibrium_eigenvalues_text = st.empty()

species = ['wolf', 'rabbit', 'grass']

num_species = len(species)

wolves_0 = st.slider('Starting wolves', 0.0, 10.0, 2.0)
rabbit_0 = st.slider('Starting rabbits', 0.0, 10.0, 4.0)
grass_0 = st.slider('Starting grass', 0.0, 10.0, 10.0)

x0 = [wolves_0, rabbit_0, grass_0]

p = np.array([x0])

r = np.array([-0.3, 0.4, 0.8])

df_p = pd.DataFrame(p, columns=species)


# interaction matrix
A = np.array([
    [    -0.1,      0.2,    0],
    [   -0.3,     -0.1,   0.1],
    [     0,     -0.2,    -0.04]
])

# dx(t)/dt = Diag(x(t)) · (r + Ax(t))
# x(t) column vector of n spieces densities at time t
# r vector intrinsic growth/death rates when alone
# A is n x n of interaction coeficients

#Diag(x(t)) · is the same as doing pair-wise multiplication

# for one single species, the solution of the ODE is
# x(t) = r / (e^(-r(k+t)) + a) = rx0e^(rt) / (r - ax0(e^(rt) - 1))
# the logistic

def glv(x, t, r, A):

     x[x < 0.00000001] = 0 # prevent numerical problems

     dxdt = x*(r + A.dot(x))

     return dxdt

equilibrium = -np.linalg.inv(A).dot(r)

st.markdown('The Jacobian at the equilibrium points is called community matrix, and for LGV is $M = J|x^* = D(x^*)·A$')

st.markdown(f"Equilibrium point: {equilibrium}")

community_matrix = equilibrium*A
eigen = np.linalg.eig(community_matrix)

eigens = eigen[0].tolist()

st.markdown("Eigenvalues:")

eigens

st.markdown("If $Re(\lambda_1) < 0 → x^*$ the equilibrium is stable")

print("")

chart = st.line_chart(df_p)

for i in range(150):
    # Update progress bar.
    progress_bar.progress((i + 1)/200)

    foo = A.dot(p[-1])
    # dx = p[-1]*(r + a.dot(p[-1]))
    # xi = dx + p[-1]
    # new_row = np.array([xi])

    new_row = p[-1] + glv(p[-1], 1, r, A)

    new_row[new_row < 0.0000001] = 0 # prevent numerical problems

    # Update status text.
    status_text.text(f'The latest population numbers are: {new_row} = {p[-1]} + {p[-1]}*({r} + {foo})')
    print(f'The latest population numbers are: {new_row} = {p[-1]} + {p[-1]}*({r} + {foo})')

    p = np.vstack([p, new_row])

    df_new_row = pd.DataFrame([new_row], columns=species)

    # Append data to the chart.
    chart.add_rows(df_new_row)

    # Pretend we're doing some computation that takes time.
    time.sleep(0.01)

# status_text.text('Done!')

chart_data = pd.DataFrame(p, columns=species)

chart_data

chart_xy = st.line_chart(chart_data, x='wolf', y='rabbit')

st.balloons()

