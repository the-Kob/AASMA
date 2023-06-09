### AASMA

### Quickstart:

1. Clone the repo or download the source code


    $ git clone https://github.com/the-Kob/AASMA
    
    $ cd AASMA

1. Create virtual environment (tested with python 3.8.10)


    $ python3 -m venv venv
    
    $ source venv/bin/activate

3. Install dependencies

    $ pip install -r requirements.txt

4. Run the project


    $ cd Project
    
    $ python3 multi_agent_teams.py

5. (Optional) Fiddle and play with the values and the teams being tested in the multi_agents_teams.py

    $ To view the ants moving around, uncommment lines 90 and 91

    $ To change the teams, change the run_multi_agent function accordingly and the n_agents in the environment definition in main, this if you decide to add more agents to the teams

    $ You can also change other aspects of the environment in main, ie. the number of foodpiles

### Code contributors:
- Carolina Brás (@carolinabras)
- Guilherme Pereira (@the_Kob)
- Miguel Belbute (@zorrocrisis)