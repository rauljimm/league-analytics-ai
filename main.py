# main.py
from views.register_view import RegisterView
from views.login_view import LoginView

class App:
    def __init__(self):
        self.register_view = None
        self.login_view = None
    
    def show_register(self):
        if self.login_view:
            self.login_view.hide()
        if not self.register_view:
            self.register_view = RegisterView(switch_to_login_callback=self.show_login)
        self.register_view.show()
    
    def show_login(self):
        if self.register_view:
            self.register_view.hide()
        if not self.login_view:
            self.login_view = LoginView(switch_to_register_callback=self.show_register)
        self.login_view.show()
    
    def run(self):
        self.show_login()
        if self.login_view:
            self.login_view.mainloop()
        elif self.register_view:
            self.register_view.mainloop()

def main():
    app = App()
    app.run()

if __name__ == "__main__":
    main()