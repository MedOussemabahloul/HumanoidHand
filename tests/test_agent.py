import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.sac_agent import SACAgent

print("Début du test : test_agent_import")
from agents.sac_agent import SACAgent

def test_agent_import():
    print("Test d'import et d'instanciation de SACAgent")
    agent = SACAgent()
    obs = [0.0, 0.0]
    action = agent.select_action(obs)
    print("Action produite par SACAgent :", action)
    print("Test SACAgent terminé avec succès.")

if __name__ == "__main__":
    test_agent_import()
