from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

keyboards = {"empty": [[]], "start": [["Начало работы"], ["Помощь"]]}


def get_keyboard(name: str, back: bool = False):
    if name not in keyboards:
        raise ValueError(f"Invalid name of keybord: {name}")
    current_keybord = []
    for key in keyboards[name]:
        current_keybord.append(
            [
                InlineKeyboardButton(text=text_key, callback_data=f"{name}:{text_key}")
                for text_key in key
            ]
        )
    if back:
        back_text = "<< Назад"
        current_keybord.append(
            [InlineKeyboardButton(text=back_text, callback_data=f"{name}:{back_text}")]
        )
    return InlineKeyboardMarkup(inline_keyboard=current_keybord)
