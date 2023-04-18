import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Attack phase: spiral towards current best solution
def spiral(m, l, X_star, X):
    E_prime = np.abs(X_star - X)
    X_next = E_prime * np.exp(m * l) * np.cos(2 * np.pi * l) + X_star
    return X_next

# Attack phase: encircle the current best solution
def encircle(a, X_star, X):
    A = 2*a*np.random.uniform(0,1, size=2) - a
    C = 2 * np.random.uniform(0,1,size=2)
    D = np.abs(C*X_star - X)
    X_next = X_star - A * D
    return X_next

# Exploration phase: move towards a random solution
def search(a, X_rand, X):
    A = 2*a*np.random.uniform(0,1, size=2) - a
    C = 2 * np.random.uniform(0,1,size=2)
    D1 = np.abs(C*X_rand - X)
    X_next = X_rand - A * D1
    return X_next   

# Redraw the plot
def set_plot(X_list, colors, X_star, ax, lim, show_previous_states, rand_idx=None, actual_solution=None):
    ax.clear()
    ax.plot(X_star[0], X_star[1], 'rx', label='X_star', markersize=10)
    
    if actual_solution is not None:
        ax.plot(actual_solution[0], actual_solution[1], 'g*',
                label='Actual Solution', markersize=10)

    for idx, (X, color) in enumerate(zip(X_list, colors)):
        if X:
            marker = 's' if idx==rand_idx else '8'
            if show_previous_states:
                X_array = np.vstack(X)
                ax.plot(X_array[:, 0], X_array[:, 1],
                        f'{marker}--',
                        label=f'X{idx+1}',
                        color=color)

            else:
                X_last = X[-1]
                ax.plot(X_last[0], X_last[1],
                            f'{marker}',
                            label=f'X{idx+1}',
                            color=color)

            
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Position of agents (X) and best solution (X_star)')
    ax.legend()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    return 


# Clip to search space constraints
def clip(X, lim):
    out = np.zeros(2)
    for i, x in enumerate(X):
        if x < -lim:
            out[i] = -lim
        elif x > lim:
            out[i] = lim
        else:
            out[i] = x
        
    return out

def update_positions_text(X_list, X_star):
    col1.text(f"X_star: {np.round(X_star, 2)}")

    for idx, X in enumerate(X_list):
        col1.text(f"X{idx+1}: {np.round(X[-1], 2)}")
        
        
def initialize(ax, lim, colors, num_agents, show_previous_states):
    initial_positions = [np.random.randint(-lim, lim, size=(2,)) for _ in range(num_agents)]
    st.session_state.initial_positions = initial_positions
    st.session_state.X_list = list(map(lambda x: [x], initial_positions))
    st.session_state.update_plot = True
    st.session_state.X_star = np.random.randint(-lim,lim,size=(2,))
    set_plot(st.session_state.X_list, colors, st.session_state.X_star, ax, lim,show_previous_states)
    return 

def calculate_a(iteration, max_iter):
    a = 2 - iteration * (2 / max_iter)
    return a

def fitness_function(X, solution):
    x, y = X
    p, q = solution
    return (x - p) ** 2 + (y - q) ** 2

def update_X_star(X_list, X_star, solution):
    best_agent = min(X_list, key=lambda x: fitness_function(x[-1], solution))
    best_fitness = fitness_function(best_agent[-1], solution)

    if fitness_function(X_star, solution) > best_fitness:
        X_star = best_agent[-1]

    return X_star





#%% Streamlit app
st.title("Whale Optimization Algorithm")


# Write formulas for WOA
st.latex(r"Spiral: \mathbf{X}_{next} = \mathbf{D} \cdot e^{m\mathbf{L}} \cdot \cos(2\pi \mathbf{L}) + \mathbf{X}^*")
st.latex(r"Encircle: \mathbf{X}_{next} = |\mathbf{X}^* - \mathbf{A} \cdot \mathbf{D}|")
st.latex(r"Search: \mathbf{X}_{next} = \mathbf{X}_{rand} - \mathbf{A} \cdot \mathbf{D_1}")

st.latex(r"\mathbf{C} = 2 \mathbf{r1}, \mathbf{D} = |\mathbf{C} \cdot \mathbf{X}^* - \mathbf{X}|,  \mathbf{L} \sim \mathcal{U}(-1, 1)^2")
st.latex(r"\mathbf{A} = 2 \cdot a \cdot \mathbf{r2} - a, a(t) = 2 - t \cdot \frac{2}{Max\_iter}")
st.latex(r"\mathbf{r1},\mathbf{r2} \sim \mathcal{U}(0, 1)^2, \mathbf{D_1} = |\mathbf{C} \cdot \mathbf{X}_{rand} - \mathbf{X}|")

# Setup streamlit
col1, col2 = st.columns([1, 3])
num_agents = st.sidebar.slider("Number of agents", 1, 10, 3)
m = st.sidebar.slider("Coefficient m", 0.1, 10.0, 0.5)
show_previous_states = st.sidebar.checkbox("Show previous states", value=False)

colors = plt.cm.jet(np.linspace(0, 1, num_agents))
fig, ax = plt.subplots()

# some constants
lim = 100 
solution = np.array([40, 30])
max_iter = 100
st.sidebar.text(f'Max_iter = {max_iter}')
st.sidebar.latex(r"Fitness(\mathbf{X}) = (x - 40)^2 + (y - 30)^2")


#%%
# initialize

if "current_iter" not in st.session_state:
    st.session_state.current_iter = 0


if "initial_positions" not in st.session_state or \
    st.session_state.prev_num_agents != num_agents or\
    st.session_state.prev_m != m or\
    st.session_state.prev_show_previous_states != show_previous_states:
        
    initialize(ax, lim, colors, num_agents, show_previous_states)
    st.session_state.prev_num_agents = num_agents
    st.session_state.prev_m = m
    st.session_state.prev_show_previous_states = show_previous_states

else:
    initial_positions = st.session_state.initial_positions
    

X_list = st.session_state.X_list
X_star = st.session_state.X_star
    
# update_positions_text(X_list, X_star)

    
if "update_plot" not in st.session_state:
    st.session_state.update_plot = False


update_funcs = [spiral, encircle, search]


if col1.button("Reset"):
    initialize(ax, lim, colors, num_agents, show_previous_states)


for btn_idx, update_func in enumerate(update_funcs):
    if col1.button(f"{update_func.__name__.capitalize()}"):
        for idx, X in enumerate(X_list):
            l = np.random.uniform(-1, 1, 2)
            a = calculate_a(st.session_state.current_iter, max_iter)
            if update_func == spiral:
                X_next = update_func(m, l, X_star, X[-1])
            elif update_func == encircle:
                X_next = update_func(a, X_star, X[-1])
            else: #search
                rand_idx = np.random.choice(range(num_agents))  # Choose a random agent index
                X_rand = X_list[rand_idx][-1]  # Select the random agent's current position as X_rand
                X_next = update_func(a, X_rand, X[-1])
                st.session_state.rand_idx = rand_idx

            X_next = clip(X_next, lim)
            X_list[idx].append(X_next)
            st.session_state["X_list"] = X_list
            st.session_state['X_star'] = X_star
        st.session_state.update_plot = True    
        st.session_state.current_iter += 1

if st.session_state.update_plot:
    if 'rand_idx' in st.session_state:  # Check if there is a rand_idx in the session state
        set_plot(st.session_state.X_list, colors, X_star, ax,
                 lim, show_previous_states,
                 rand_idx=st.session_state.rand_idx,
                 actual_solution = solution)
        del st.session_state.rand_idx  # Delete rand_idx from the session state after plotting
    else:
        set_plot(st.session_state.X_list, colors, st.session_state.X_star,
                 ax, lim, show_previous_states,
                 actual_solution = solution)
   
    st.session_state.update_plot = False
    
if col1.button('update x_star'):
    X_star = update_X_star(X_list, X_star, solution)
    st.session_state['X_star'] = X_star
    set_plot(st.session_state.X_list, colors, st.session_state.X_star,
             ax, lim,
             show_previous_states,
             actual_solution = solution)





col2.pyplot(fig)


  
