print("Import de utils/logger.py réussi.")

class Logger:
    def __init__(self):
        print("Logger initialisé.")

    def log(self, msg):
        print(f"[LOG] {msg}")

logger_instance = Logger()
log = logger_instance.log
