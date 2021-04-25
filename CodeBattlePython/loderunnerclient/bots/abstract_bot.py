import uuid
import time


class AbstractBot:
    def __init__(self):
        self.uuid = str(uuid.uuid4())
        self.start_time = None
        # время в мс, когда мы должны прервать вычисление экшена и вернуть хотя бы рандом
        self.emergency_exit_time = 900

    def make_action(self):
        self.start_time = self.current_time()
        return self.choose_action()

    def choose_action(self):
        # have to be implemented in bot
        pass

    def elapsed_time(self):
        if self.start_time is not None:
            return self.current_time() - self.start_time
        raise RuntimeError("Cant eval time: start time was not set")

    def current_time(self):
        return round(time.time() * 1000)
