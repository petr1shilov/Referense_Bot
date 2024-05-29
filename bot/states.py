from aiogram.fsm.state import State, StatesGroup

class UserStates(StatesGroup):
    get_answer = State()
    get_query = State()
    get_pdf = State()
    get_params = State()