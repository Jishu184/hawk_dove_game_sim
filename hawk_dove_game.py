


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import threading
import numpy.random as nr
import numpy as np

# CSS styling








# internal functions 





def a_eq(pl1_hh, pl1_hd, pl1_dh, pl1_dd, pl2_hh, pl2_hd, pl2_dh, pl2_dd):
    """
    Calculates the mixed strategy Nash equilibrium for an asymmetric Hawk-Dove game.
    
    Parameters:
        payoff_matrix: 2x2 numpy array containing payoffs.
                      Format: [[(a1, a2), (b1, c2)], [(c1, b2), (d1, d2)]]
    
    Returns:
        p: Probability Player 1 plays Hawk.
        q: Probability Player 2 plays Hawk.
    """
    # Extract payoffs
    x2=(pl2_hh-pl2_hd-pl2_dh+pl2_dd)
    x1=(pl1_hh-pl1_hd-pl1_dh+pl1_dd)
    if x1==0:
        st.markdown('All the strategies for player 1 are equilibrium strategy')
    if x2==0:
        st.markdown('All the strategies for player 2 are equilibrium strategy')
    if x1*x2 !=0:
        # Calculate probabilities
        p = (pl2_dd-pl2_hd)/(pl2_hh-pl2_hd-pl2_dh+pl2_dd) # Player 1's probability of playing Hawk
        q = (pl1_dd-pl1_hd)/(pl1_hh-pl1_hd-pl1_dh+pl1_dd)  # Player 2's probability of playing Hawk
        if p > 1:
            p=1
        elif p<0:
            p=0
        if q > 1:
            q=1
        elif q<0:
            q=0  

        st.markdown(f'Equilibrium strategy for Player 1, $p_{{eq}}$ :   {p}')
        st.markdown(f'Equilibrium strategy for player 2 , $q_{{eq}}$ : {q}') 

# Assuming the function e and ber are defined somewhere

        

def ber(p):
    u=nr.random()
    if u<p:
        return 1
    else:
        return 0

def plot_gradual_evolution(n, x0, p, r, m, k,f, x_inf):
    # Initialize list for x
    x = [x0]
    n1 = []

    # Create a placeholder for the plot
    plot_placeholder = st.empty()

    # Gradually generate data for x and plot in Streamlit
    for t in range(k):
        a = [0 for i in range(n)]
        count = [0 for i in range(n)]
        avg_pay = [0 for i in range(n)]
        n1.append(round(n * x[t]))
        
        for i in range(m * n):  # Number of conflicts
            p1 = nr.randint(0, n-1)
            p2 = nr.randint(0, n-1)
            count[p1] += 1
            count[p2] += 1
            if p1 == p2:
                count[p1] -= 1
                count[p2] -= 1
            elif (p1 < n1[t]) and (p2 < n1[t]):
                s1 = ber(r)
                s2 = ber(r)
                a[p1] += e(s1, s2)
                a[p2] += e(s2, s1)
            elif (p1 < n1[t]) and (p2 >= n1[t]):
                s1 = ber(r)
                s2 = ber(p)
                a[p1] += e(s1, s2)
                a[p2] += e(s2, s1)
            elif (p1 >= n1[t]) and (p2 >= n1[t]):
                s1 = ber(p)
                s2 = ber(p)
                a[p1] += e(s1, s2)
                a[p2] += e(s2, s1)
            elif (p1 >= n1[t]) and (p2 < n1[t]):
                s1 = ber(p)
                s2 = ber(r)
                a[p1] += e(s1, s2)
                a[p2] += e(s2, s1)

        # Calculate average payoff for each player
        avg_pay = [a[i] / count[i] if count[i] > 0 else 0 for i in range(n)]
        
        # Mean of payoffs for the two strategies
        mut_avg = np.mean(avg_pay[:int(n1[t])]) if len(avg_pay[:int(n1[t])]) > 0 else 0
        non_mut_avg = np.mean(avg_pay[int(n1[t]):]) if len(avg_pay[int(n1[t]):]) > 0 else 0
        
        # Update the value of x using the evolution equation
        x_new = (f + x[t] * mut_avg) / (f + (x[t] * mut_avg) + ((1 - x[t]) * non_mut_avg))
        x.append(x_new)

        # Gradually plot the values of x
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(x)),x, marker='o', color='b', label="Mutant proportion")
        ax.axhline(y=x_inf, color='r', linestyle='--', label='$x_{inf}$')
        ax.axhline(y=x_inf+0.01, color='blue', linestyle='--', label='$x_{inf}+.01$')
        ax.axhline(y=x_inf-0.01, color='blue', linestyle='--', label='$x_{inf}-.01$')
        plt.fill_between(range(len(x)), x_inf-0.01, x_inf+0.01, color='skyblue', alpha=0.5)
        ax.set_title("Gradual Evolution of mutant population")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mutant Proportion")
        ax.grid(True)
        ax.legend()

        # Update the plot in Streamlit
        plot_placeholder.pyplot(fig)

        # Simulate delay before next update
        time.sleep(.1)

        # Close the figure after each update to avoid overlapping
        plt.close()



           
            



















# buttons variables 

if 'btn1' not in st.session_state:
    st.session_state['btn1']=0

if 'btn2' not in st.session_state:
    st.session_state['btn2']=0


def btn1():
    if st.session_state['btn1']==0:
        st.session_state['btn1']=1
    else:
        st.session_state['btn1']=0

def btn2():
    if st.session_state['btn2']==0:
        st.session_state['btn2']=1
    else:
        st.session_state['btn2']=0


# buttons variables 












# Title of the app
st.title("Hawk-Dove Game Simulator")

# Sidebar panel for other inputs
st.sidebar.header("Parameters")
p = st.sidebar.slider("Majority Strategy (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
r = st.sidebar.slider("Mutant Strategy (r)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
x0 = st.sidebar.slider("Initial Mutant proportion ($x_0$)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
# n = st.sidebar.slider("Number of creatures ($n$)", min_value=0, max_value=5000, value=2000, step=100)
# m = st.sidebar.slider("Number of conflicts : multiplier of $n$   $(m)$  ", min_value=100, max_value=1000, value=100, step=100)
k = st.sidebar.slider("Number of generations", min_value=100, max_value=1000, value=100, step=100)
# n_conflict=n*m
# Create tabs
tab1, tab2 ,tab3= st.tabs(["Pay-off Matrix","Theoretical Study", "Simulation Study"])

# Tab 1: Input payoff matrix and display it
with tab1:
    st.subheader("Input Payoff Matrix")

    # Input for payoff matrix using two columns
    col1, col2 , col3 , col4 = st.columns(4)
    with col1:
        hawk_hawk_p1 = st.number_input("Hawk vs Hawk (P1)", value=-5.0, step=0.1)
        hawk_hawk_p2 = st.number_input("Hawk vs Hawk (P2)", value=-5.0, step=0.1)

    with col2:
        hawk_dove_p1 = st.number_input("Hawk vs Dove (P1)", value=10.0, step=0.1)
        hawk_dove_p2 = st.number_input("Hawk vs Dove (P2)", value=10.0, step=0.1)
    with col3:
        dove_hawk_p1 = st.number_input("Dove vs Hawk (P1)", value=0.0, step=0.1)
        dove_hawk_p2 = st.number_input("Dove vs Hawk (P2)", value=0.0, step=0.1)

    with col4:
        dove_dove_p1 = st.number_input("Dove vs Dove (P1)", value=5.0, step=0.1)
        dove_dove_p2 = st.number_input("Dove vs Dove (P2)", value=5.0, step=0.1)





        

    # Create and display the payoff matrix
    payoff_matrix = pd.DataFrame(
        [
            [(hawk_hawk_p1, hawk_hawk_p2), (hawk_dove_p1, hawk_dove_p2)],
            [(dove_hawk_p1, dove_hawk_p2), (dove_dove_p1, dove_dove_p2)],
        ],
        columns=["Player 2: Hawk", "Player 2: Dove"],
        index=["Player 1: Hawk", "Player 1: Dove"],
    )

    st.subheader("Payoff Matrix")
    st.table(payoff_matrix)
    st.markdown("---")
    # Add a button to simulate the game



# Tab 2: Display a message
with tab2:

    a_eq(hawk_hawk_p1, hawk_dove_p1, dove_hawk_p1, dove_dove_p1, hawk_hawk_p2, hawk_dove_p2, dove_hawk_p2, dove_dove_p2)
    st.markdown("---")
    st.markdown("### Evolutionary Stability of majority population")
    st.markdown("""
    * ESS condition 
        * $x_{t}\cdot[E(p,p)-E(r,p)]-(1-x_{t})\cdot[E(p,r)-E(r,r)]>0$
    """)
    st.markdown(" ")
    



    def e(i,j,pl=1):
        if pl==1:                
            if i==0 and j==0:
                return dove_dove_p1  # dd
            elif i==1 and j==0:
                return  hawk_dove_p1 # hd
            elif i==0 and j==1:
                return  dove_hawk_p1      #  dh
            elif i==1 and j==1 :
                return  hawk_hawk_p1   # hh
        if pl==2:                
            if i==0 and j==0:
                return dove_dove_p2  # dd
            elif i==1 and j==0:
                return  hawk_dove_p2 # hd
            elif i==0 and j==1:
                return  dove_hawk_p2      #  dh
            elif i==1 and j==1 :
                return  hawk_hawk_p2   # hh


    def em(p,q,pl=1):
        if pl==1:
            a11=p*q*e(1,1,1)
            a10=p*(1-q)*e(1,0,1)
            a01=(1-p)*q*e(0,1,1)
            a00=(1-p)*(1-q)*e(0,0,1)
            b=a11+a10+a01+a00
            return b
        elif pl==2:
            a11=p*q*e(1,1,2)
            a10=p*(1-q)*e(1,0,2)
            a01=(1-p)*q*e(0,1,2)
            a00=(1-p)*(1-q)*e(0,0,2)
            b=a11+a10+a01+a00
            return b  
    ess_val=(1-x0)*(em(p,p)-em(r,p))+x0*(em(p,r)-em(r,r))  
    if ess_val >0:
        st.markdown(f"""
        * In current situation ,
            * $c_{{t}}=x_{{t}}\cdot[E(p,p)-E(r,p)]-(1-x_{{t}})\cdot[E(p,r)-E(r,r)]   =  {round(ess_val,3)} >0$
        """)
        st.markdown("* Comments ")
        st.markdown("So the majority strategy is evolutionary stable ")
        st.markdown('Mutant proportion will tend to decrease  ')         
    elif ess_val<0:
        st.markdown(f"""
        * In current situation ,
            * $c_{{t}}=x_{{t}}\cdot[E(p,p)-E(r,p)]-(1-x_{{t}})\cdot[E(p,r)-E(r,r)]   =  {round(ess_val,3)} <0$
        """)
        st.markdown("* Comments ")
        st.markdown("So the majority strategy is NOT evolutionary stable ")
        st.markdown('Mutant proportion  will tend to increase   ')
    elif ess_val==0:
        st.markdown(f"""
        * In current situation ,
            * $c_{{t}}=x_{{t}}\cdot[E(p,p)-E(r,p)]-(1-x_{{t}})\cdot[E(p,r)-E(r,r)]   =  {round(ess_val,3)}$
        """)
        st.markdown("* Comments ")
        st.markdown("So the majority strategy is NOT evolutionary stable ")
        st.markdown(f'Mutant proportion  will tend to move around $x_0 = {x0}$   ')   

    st.markdown("---")
    if abs(ess_val)<.01 :
       
        st.markdown("NOTE: ")
        st.markdown(f'* $c_{{t}}$ = {round(ess_val, 4)} is very close to 0 ')
        st.markdown(f'* The mutant proportion will very slowly move from initial proportion $x_{0}= {round(x0,4)}$')
        st.markdown("---")
    a=(em(p,p)-em(r,p))
    b=em(p,r)-em(r,r)
    if a!=b:
        x_inf=a/(a-b)
        if x_inf>=1:
            x_inf=1
        elif x_inf<=0:
            x_inf=0
        st.markdown("### Evolutionary equilibrium proportion")
        st.markdown(f'* Evolutionary equilibrium proportion for mutant $(x_{{inf}}) = {round(x_inf,4)}$ ')
        st.markdown(f"* The mutant population proportion eventually converge to $(x_{{inf}}) = {round(x_inf,4)}$")
    
with tab3:
    sim_work = st.selectbox(
    '',
    ['Simulation Overview', 'Simulation Results']
        )
    if  sim_work==    'Simulation Overview':
        st.markdown('### Simulation Overview')
        st.markdown(f"""
        * Suppose an island can accommodate $n$ birds ($n$ is set to a large number). We assume there is no inflow or outflow of birds in this population.
        * Two species of birds are there .
            * Species A and B .
            * A is the majority species .
            * B is mutant Species.
        * In the 1st generation ($0^{{th}}$ generation ) , mutant proportion  $(x_{{0}})={x0}$
        * In each generaion there will be one-to-one conflicts between the birds for food. 
            * Number of conflicts in each generation is a multiple of $n$
            * Let the multiplier $m$
            * Total number of conflicts in each generation  $(n_{{conflict}})=m\cdot n$
        * Number of simulated generations $(k)={k}$
        * Rule for one-to-one conflict 
            * For each conflict two distinct birds will be rabndomly choosen from the population.
            * Majority strategy $(p)={p}$
            * Mutant  strategy $(r)={r}$
            * Each species A bird  will randomly select a strategy (Hawk strategy with probability $p={p}$ )
            * Each species B bird  will randomly select a strategy (Hawk strategy with probability $r={r}$ )
            * There is no limit on the maximum number of conflicts a bird can face in its lifetime. 
            * Total fitness of a bird is equal to the sum of basic fitness and the payoff in the fame.
            * The growth factors of the mutant population are proportional to total fitness of r.
        * New generation simulation
            * At the end of any generation, the birds will give birth to offsprings and will die . 
            * If the avg. Reproductive capacity of mutant population is more than that of majority population , mutant population will increase .
            """)
    if sim_work=='Simulation Results':
        b1=st.button('Start Simulation')
        if b1:                
            col1, col2=st.columns([4, 8])
            with col1:
                st.markdown("### Simulation 1")
                st.markdown("* $n= 100 , m=100$")
                st.markdown(f"""
                    * Slower convergence rate due to small population size and less number of conflicts 
                    * Red horisontal line is the ling term evolutionary equilibrium proportion 
                    * Eventually the proportion sequence  will enter the blue region and tend stay there
                    * Equilibrium proportion $x_{{inf}}={x_inf}$


                    """)
            with col2:
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                plot_gradual_evolution(n=100, x0=x0, p=p, r=r, m=100, k=k,f=0, x_inf=x_inf)
            
            st.markdown('---')
            with col1: 
                st.markdown('---')
                st.markdown("### Simulation 2")
                st.markdown("* $n= 500 , m=250$")
                st.markdown(f"""
                    * Moderate convergence rate due to increased population size and increased number of conflicts 
                    * Lesser randomness but simulation is slower
                    * Faster convergence 
                    * Equilibrium proportion $x_{{inf}}={x_inf}$


                    """)
            with col2:
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                plot_gradual_evolution(n=500, x0=x0, p=p, r=r, m=500, k=k,f=0, x_inf=x_inf)            
            st.markdown('---')
            with col1: 

                st.markdown("### Simulation 3")
                st.markdown("* $n= 1000 , m=500$")
                st.markdown(f"""
                    * Moderate convergence rate due to greater population size and greater number of conflicts 
                    * Time complexity is more
                    * Will enter the blue band faster 
                    * Equilibrium proportion $x_{{inf}}={x_inf}$


                    """)
            with col2:
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                plot_gradual_evolution(n=1000, x0=x0, p=p, r=r, m=500, k=k,f=0, x_inf=x_inf) 



















 # streamlit run '/Users/jisuadhikary/Documents/python docs/hawk_dove_game_simulator.py'
