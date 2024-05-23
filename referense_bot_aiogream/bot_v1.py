import argparse
import asyncio
import base64
import fitz
import os
import time


from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.state import default_state
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    InputFile,
    CallbackQuery,
    ErrorEvent,
    InputSticker,
    Message,
    ReplyKeyboardRemove,
    ContentType,
    FSInputFile,
)
from aiogram.utils.deep_linking import create_start_link

import config

from bot.keyboards import get_keyboard
from bot.states import UserStates
from bot.texts import *
from api import AnswerAPI

TOKEN = config.test_api_key

# Нужно для поднятия локальной базы, что бы созранять передменные
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
bot = Bot(TOKEN)


@dp.message(Command("help"))
async def command_help_handler(message: Message, state: FSMContext) -> None:
    await message.answer(help_message_text)


# @dp.message(UserStates.processing)
# async def processing_handler(message: Message, state: FSMContext) -> None:
#     await message.answer(help_message_text) #processing_message_text)


@dp.message(CommandStart())
async def command_start_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(UserStates.get_params)
    await message.answer(hello_message_text, reply_markup=ReplyKeyboardRemove())
    await message.answer(start_message_text)
    await message.answer(pdf_message_text)
    messege_id = message.message_id
    user_id = message.from_user.id
    await state.update_data(delete_messege=[messege_id + 3], user_id=user_id)
    await state.set_state(UserStates.get_pdf)


@dp.message(UserStates.get_pdf, F.content_type == "document")
async def get_pdf_handler(message: Message, state: FSMContext):

    data = await state.get_data()
    message_id = data["delete_messege"]
    await bot.delete_messages(chat_id=message.chat.id, message_ids=message_id)
    user_data = await state.get_data()
    user_id = user_data["user_id"]

    file_id = message.document.file_id
    file_name = f"{str(user_id)}_{message.document.file_name}"
    await state.update_data(file_name=file_name)

    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, f"files/{file_name}")
    await message.answer(query_message_text, reply_markup=get_keyboard("empty"))
    messege_id = message.message_id
    await state.update_data(delete_messege=[messege_id + 1])
    await state.set_state(UserStates.get_query)


@dp.message(StateFilter(UserStates.get_pdf), F.content_type != "document")
async def warning_not_pdf(message: Message, state: FSMContext):
    data = await state.get_data()
    message_id = data["delete_messege"]
    message_id.append(message.message_id - 1)

    await bot.delete_messages(chat_id=message.chat.id, message_ids=message_id)
    answer_text = f"{warning_pdf_message}\n\n{pdf_message_text}"
    await message.answer(text=answer_text)
    messege_id = message.message_id
    await state.update_data(delete_messege=[messege_id, messege_id + 1])
    data = await state.get_data()
    message_id = data["delete_messege"]  # можно улучшить удаление файлов


# @dp.callback_query(UserStates.get_query, F.data.startswith('empty'))
# async def callback_query_back_handler(query: CallbackQuery, state: FSMContext) -> None:
#     await state.set_state(UserStates.get_params)
#     await bot.edit_message_text(chat_id=query.message.chat.id, message_id=query.message.message_id,
#                                         text=pdf_message_text, reply_markup=get_keyboard('empty', back=True))
#     await state.set_state(UserStates.get_pdf)


@dp.message(UserStates.get_query, F.content_type == "text")
async def get_query_handler(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    message_id = data["delete_messege"]
    await bot.delete_messages(chat_id=message.chat.id, message_ids=message_id)
    request = message.text.strip()
    await state.update_data(request=request)
    await send_file(message, state)


@dp.message(UserStates.get_query, F.content_type != "text")
async def warning_not_query(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    message_id = data["delete_messege"]
    message_id.append(
        message.message_id - 1
    )  # тут можно поиграть с id и убрать этот append или на 100 строке

    await bot.delete_messages(chat_id=message.chat.id, message_ids=message_id)
    answer_text = f"{warning_query_message}\n\n{query_message_text}"
    await message.answer(text=answer_text)
    messege_id = message.message_id
    await state.update_data(delete_messege=[messege_id, messege_id + 1])
    data = await state.get_data()
    message_id = data["delete_messege"]


async def send_file(message: Message, state: FSMContext) -> None:
    user_data = await state.get_data()
    file_name = user_data["file_name"]
    request = user_data["request"]
    path = f"files/{file_name[:-4]}_modified.pdf"

    msg = await message.answer(waiting_message)

    test = AnswerAPI(
        file_name, request
    )  # придумать как лучше ввести изменение параметров
    test.get_modified_file()

    while not os.path.exists(path=path):
        time.sleep(0.5)
    message_id = msg.message_id
    await bot.delete_messages(chat_id=message.chat.id, message_ids=[message_id])
    await message.answer_document(FSInputFile(path))
    # доделать кнопки рефреш и кнопку для ввода нового запроса


async def delete_pdf(message: Message, state: FSMContext) -> None:
    user_data = await state.get_data()
    file_name = user_data["file_name"]
    path = f"/Users/petrsilov/Desktop/LinkValidationSystem/bot_v2/files/{file_name}"
    if os.path.exists(path=path):
        os.remove(path=path)


@dp.message(StateFilter(default_state))
async def send_echo(message: Message):
    await message.reply(text="Извините, моя твоя не понимать")


if __name__ == "__main__":
    # logger.debug('Start polling')
    asyncio.run(dp.start_polling(bot))