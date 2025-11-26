import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Draw Agent Box
    agent_box = patches.FancyBboxPatch((1, 2), 2, 2, boxstyle="round,pad=0.1", ec="blue", fc="lightblue")
    ax.add_patch(agent_box)
    ax.text(2, 3, "Agent\n(Drone)", ha="center", va="center", fontsize=12, fontweight='bold')
    
    # Draw Environment Box
    env_box = patches.FancyBboxPatch((7, 2), 2, 2, boxstyle="round,pad=0.1", ec="green", fc="lightgreen")
    ax.add_patch(env_box)
    ax.text(8, 3, "Environment\n(City Grid)", ha="center", va="center", fontsize=12, fontweight='bold')
    
    # Arrows
    # Action Arrow (Agent -> Env)
    ax.annotate("", xy=(7, 3.5), xytext=(3, 3.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(5, 3.7, "Action (Move/Interact)", ha="center", fontsize=10)
    
    # State/Reward Arrow (Env -> Agent)
    ax.annotate("", xy=(3, 2.5), xytext=(7, 2.5), arrowprops=dict(arrowstyle="->", lw=2, ls="--"))
    ax.text(5, 2.7, "State (Pos, Battery) + Reward", ha="center", fontsize=10)
    
    # Description
    desc = (
        "RL Interaction Loop:\n"
        "1. Agent observes state (Position, Battery, Wind)\n"
        "2. Agent selects action (Move, Ascend, Hover)\n"
        "3. Environment updates state based on physics\n"
        "4. Environment returns new state and reward"
    )
    ax.text(5, 0.5, desc, ha="center", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title("Agent-Environment Interaction Diagram", fontsize=14)
    plt.tight_layout()
    plt.savefig("analysis/agent_environment_diagram.png")
    print("Saved diagram to analysis/agent_environment_diagram.png")

if __name__ == "__main__":
    create_diagram()
