import streamlit as st
import numpy as np
import pandas as pd
import time

progress_bar = st.progress(0)
status_text = st.empty()

species = ['wolf', 'rabbit', 'grass']

num_species = len(species)

x0 = [0.5, 0.5, 0.5]

p = np.array([x0])

r = np.array([-0.2, 0.1, 0])

chart = st.line_chart(p)

# interaction matrix
a = np.array([
    [0,0.2,0],
    [-0.3,0,0],
    [0,0,0]
])

for i in range(20):
    # Update progress bar.
    progress_bar.progress(i + 1)

    for j in range(10):
        foo = a.dot(p[-1])
        dx = p[-1]*(r + a.dot(p[-1]))
        xi = dx + p[-1]
        new_row = np.array([xi])
        p = np.vstack([p, xi])

        # Update status text.
        status_text.text(
            'The latest population numbers are: %s' % foo)

        # Append data to the chart.
        chart.add_rows(new_row)

    # Pretend we're doing some computation that takes time.
    time.sleep(0.1)

status_text.text('Done!')

p

st.balloons()

