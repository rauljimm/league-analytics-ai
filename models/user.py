class User:
    username: str
    email: str
    password: str

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password
    