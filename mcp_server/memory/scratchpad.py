import time


class Scratchpad:
    def __init__(self):
        self.plan = None
        self.sub_goal = None
        self.state = {}
        self.updated_at = None

    def set_plan(self, plan: str):
        self.plan = plan
        self.updated_at = time.time()

    def set_sub_goal(self, sub_goal: str):
        self.sub_goal = sub_goal
        self.updated_at = time.time()

    def update_state(self, key: str, value):
        self.state[key] = value
        self.updated_at = time.time()

    def get_state(self, key: str, default=None):
        return self.state.get(key, default)

    def snapshot(self):
        return {
            "plan": self.plan,
            "sub_goal": self.sub_goal,
            "state": dict(self.state),
            "updated_at": self.updated_at
        }

    def clear(self):
        self.plan = None
        self.sub_goal = None
        self.state = {}
        self.updated_at = time.time()
