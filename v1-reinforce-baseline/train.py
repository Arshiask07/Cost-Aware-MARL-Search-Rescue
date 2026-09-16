from config import GAMMA

def update_policy(drone):
    G = 0
    loss = 0

    for log_prob, reward in zip(
        reversed(drone.log_probs),
        reversed(drone.rewards)
    ):
        G = reward + GAMMA * G
        loss += -log_prob * G

    drone.optimizer.zero_grad()
    loss.backward()
    drone.optimizer.step()

    drone.log_probs.clear()
    drone.rewards.clear()
