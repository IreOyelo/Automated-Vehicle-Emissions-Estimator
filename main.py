import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.gui_main import App

if __name__ == "__main__":
    app = App()
    app.mainloop()