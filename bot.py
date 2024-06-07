import logging
import os
import asyncio
import joblib
import pandas as pd
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

load_dotenv()

token = os.getenv('TOKEN_API')

logging.basicConfig(level=logging.INFO)

bot = Bot(token=token)
dp = Dispatcher()


async def reset_webhook():
    await bot.delete_webhook()


class TestStates(StatesGroup):
    Q1 = State()
    Q2 = State()
    Q3 = State()
    Q4 = State()
    FIRST_TEST = State()
    SECOND_TEST = State()
    THIRD_TEST = State()


models_dict = {}

user_data = pd.DataFrame(
    columns=['UserID', 'Education', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
             'Alcohol', 'Caff', 'Choc', 'Nicotine'])


def unpack_models():
    global models_dict
    models_dict = joblib.load('models.joblib')


education_col = {
    'Left School Before 16 years': -2.43591,
    'Left School at 16 years': -1.73790,
    'Left School at 17 years': -1.43719,
    'Left School at 18 years': -1.22751,
    'Some College, No Certificate Or Degree': -0.61113,
    'Professional Certificate/Diploma': -0.05921,
    'University Degree': 0.45468,
    'Masters Degree': 1.16365,
    'Doctorate Degree': 1.98437,
}

ethnicity_col = {
    'Asian': -0.50212,
    'Black': -1.10702,
    'Mixed-Black/Asian': 1.90725,
    'Mixed-White/Asian': 0.12600,
    'Mixed-White/Black': -0.22166,
    'Other': 0.11440,
    'White': -0.31685
}


async def normalize_data():
    global user_data
    user_data['Education'] = user_data['Education'].replace(education_col)
    user_data['Ethnicity'] = user_data['Ethnicity'].replace(ethnicity_col)
    global models_dict
    scaler = models_dict['scaler']
    user_data.iloc[:, 1:] = scaler.transform(user_data.iloc[:, 1:])


async def predict(user_id_to_find: int) -> dict:
    ans = {}
    global models_dict
    for test, model in models_dict.items():
        if test != 'scaler':
            ans[test] = model.predict_proba(
                user_data.loc[user_data['UserID'] == user_id_to_find].drop('UserID', axis=1))
    return ans


async def main():
    await reset_webhook()

    router = Router()

    start_keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text='Начать тест')]
        ],
        resize_keyboard=True
    )

    education_keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text='Left School Before 16 years')],
            [KeyboardButton(text='Left School at 16 years')],
            [KeyboardButton(text='Left School at 17 years')],
            [KeyboardButton(text='Left School at 18 years')],
            [KeyboardButton(text='Some College, No Certificate Or Degree')],
            [KeyboardButton(text='Professional Certificate/Diploma')],
            [KeyboardButton(text='University Degree')],
            [KeyboardButton(text='Masters Degree')],
            [KeyboardButton(text='Doctorate Degree')]
        ],
        resize_keyboard=True
    )

    skin_color_keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text='Asian')],
            [KeyboardButton(text='Black')],
            [KeyboardButton(text='Mixed-Black/Asian')],
            [KeyboardButton(text='Mixed-White/Asian')],
            [KeyboardButton(text='Mixed-White/Black')],
            [KeyboardButton(text='Other')],
            [KeyboardButton(text='White')]
        ],
        resize_keyboard=True
    )

    yes_no_keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text='Да')],
            [KeyboardButton(text='Нет')]
        ],
        resize_keyboard=True
    )

    test_link1 = 'https://www.truity.com/test/big-five-personality-test'
    test_link2 = 'https://qxmd.com/calculate/calculator_854/barratt-impulsiveness-scale-bis-11#'
    test_link3 = 'https://psytests.org/multi/zkpqccen-run.html'

    @router.message(Command(commands=['start']))
    async def start_command(message: types.Message):
        await message.answer(
            'Привет. Пройди небольщой опрос и по итогу ты узнаешь вероятность того, что ты когда нибудь попробуешь что то из самых популярных наркотиков(или уже попробовал))',
            reply_markup=start_keyboard)

    @router.message(lambda message: message.text == 'Начать тест')
    async def choose_education_level(message: types.Message):
        await message.answer('Выберите свой уровень образования:', reply_markup=education_keyboard)

    @router.message(lambda message: message.text in [
        'Left School Before 16 years', 'Left School at 16 years', 'Left School at 17 years',
        'Left School at 18 years', 'Some College, No Certificate Or Degree',
        'Professional Certificate/Diploma', 'University Degree', 'Masters Degree', 'Doctorate Degree'
    ])
    async def handle_education_choice(message: types.Message):
        user_id = message.from_user.id
        education_level = message.text

        global user_data
        if user_id in user_data['UserID'].values:
            user_data.loc[user_data['UserID'] == user_id, 'Education'] = education_level
        else:
            new_entry = pd.DataFrame({'UserID': [user_id], 'Education': [education_level]})
            user_data = pd.concat([user_data, new_entry], ignore_index=True)

        await message.answer('Теперь выберите рассу:', reply_markup=skin_color_keyboard)

    @router.message(lambda message: message.text in [
        'Asian', 'Black', 'Mixed-Black/Asian', 'Mixed-White/Asian', 'Mixed-White/Black', 'Other', 'White'
    ])
    async def handle_skin_color_choice(message: types.Message, state: FSMContext):
        user_id = message.from_user.id
        skin_color = message.text

        global user_data
        if user_id in user_data['UserID'].values:
            user_data.loc[user_data['UserID'] == user_id, 'Ethnicity'] = skin_color
        else:
            new_entry = pd.DataFrame({'UserID': [user_id], 'Ethnicity': [skin_color]})
            user_data = pd.concat([user_data, new_entry], ignore_index=True)

        education_level = user_data.loc[user_data['UserID'] == user_id, "Education"].values[0]

        link = f"[тест]({test_link1})"

        await message.answer(
            f'Ваш уровень образования и цвет кожи сохранены. Уровень образования: {education_level}, Цвет кожи: {skin_color}\n'
            f'Пожалуйста, пройдите первый [тест]({test_link1}) и отправьте результат в формате:\n'
            'O `<your_answer>`\nC `<your_answer>`\nE `<your_answer>`\nA `<your_answer>`\nN `<your_answer>`\n округляй значениея до целых',
            reply_markup=ReplyKeyboardRemove(),
            parse_mode="Markdown"
        )
        await state.set_state(TestStates.FIRST_TEST)

    @router.message(TestStates.FIRST_TEST)
    async def handle_test_result(message: types.Message, state: FSMContext):
        user_id = message.from_user.id

        try:
            test_results = message.text.split('\n')

            if not all(line[0] in 'OCEAN' and line[1:].strip().isdigit() for line in test_results):
                raise ValueError

            global user_data
            for result in test_results:
                trait, score = result.split()
                column_name = f'{trait}score'
                score = int(score)

                if user_id in user_data['UserID'].values:
                    user_data.loc[user_data['UserID'] == user_id, column_name] = score
                else:
                    new_entry = pd.DataFrame({'UserID': [user_id], column_name: [score]})
                    user_data = pd.concat([user_data, new_entry], ignore_index=True)

            link = f"[тест]({test_link2})"

            await message.answer(
                f'Теперь пройдите второй {link} и отправьте результат в формате:\n`<your_answer>`',
                parse_mode="Markdown"
            )
            await state.set_state(TestStates.SECOND_TEST)

        except ValueError:
            await message.answer(
                'Ответьте как в примере:\nO <your_answer>\nC <your_answer>\nE <your_answer>\nA <your_answer>\nN <your_answer>')

    @router.message(TestStates.SECOND_TEST)
    async def handle_second_test_result(message: types.Message, state: FSMContext):
        user_id = message.from_user.id

        try:
            impulsive_score = int(message.text)

            global user_data
            if user_id in user_data['UserID'].values:
                user_data.loc[user_data['UserID'] == user_id, 'Impulsive'] = impulsive_score
            else:
                new_entry = pd.DataFrame({'UserID': [user_id], 'Impulsive': [impulsive_score]})
                user_data = pd.concat([user_data, new_entry], ignore_index=True)

            link = f"[тест]({test_link3})"

            await message.answer(
                f'Теперь пройдите третий {link} и отправьте результат в формате:\n`<your_answer>`\nсмотри первую шкалу (ImpSS)',
                parse_mode="Markdown"
            )
            await state.set_state(TestStates.THIRD_TEST)

        except ValueError:
            await message.answer('Ответьте как в примере: <your_answer>')

    @router.message(TestStates.THIRD_TEST)
    async def handle_third_test_result(message: types.Message, state: FSMContext):
        user_id = message.from_user.id
        try:
            ss_score = int(message.text)

            global user_data
            if user_id in user_data['UserID'].values:
                user_data.loc[user_data['UserID'] == user_id, 'SS'] = ss_score
            else:
                new_entry = pd.DataFrame({'UserID': [user_id], 'SS': [ss_score]})
                user_data = pd.concat([user_data, new_entry], ignore_index=True)

            await message.answer('Вы пробовали когда-либо алкоголь?', reply_markup=yes_no_keyboard)
            await state.set_state(TestStates.Q1)

        except ValueError:
            await message.answer('Ответьте как в примере: 8')

    @router.message(TestStates.Q1, lambda message: message.text.lower() in ['да', 'нет'])
    async def handle_alcohol(message: types.Message, state: FSMContext):
        user_id = message.from_user.id
        answer = 2 if message.text.lower() == 'да' else 0

        global user_data
        if user_id in user_data['UserID'].values:
            user_data.loc[user_data['UserID'] == user_id, 'Alcohol'] = answer
        else:
            new_entry = pd.DataFrame({'UserID': [user_id], 'Alcohol': [answer]})
            user_data = pd.concat([user_data, new_entry], ignore_index=True)

        await message.answer('Вы пробовали когда-либо никотин?', reply_markup=yes_no_keyboard)
        await state.set_state(TestStates.Q2)

    @router.message(TestStates.Q2, lambda message: message.text.lower() in ['да', 'нет'])
    async def handle_nicotine(message: types.Message, state: FSMContext):
        user_id = message.from_user.id
        answer = 2 if message.text.lower() == 'да' else 0

        global user_data
        if user_id in user_data['UserID'].values:
            user_data.loc[user_data['UserID'] == user_id, 'Nicotine'] = answer
        else:
            new_entry = pd.DataFrame({'UserID': [user_id], 'Nicotine': [answer]})
            user_data = pd.concat([user_data, new_entry], ignore_index=True)

        await message.answer('Вы пробовали когда-либо шоколад?', reply_markup=yes_no_keyboard)
        await state.set_state(TestStates.Q3)

    @router.message(TestStates.Q3, lambda message: message.text.lower() in ['да', 'нет'])
    async def handle_choc(message: types.Message, state: FSMContext):
        user_id = message.from_user.id
        answer = 1 if message.text.lower() == 'да' else 0

        global user_data
        if user_id in user_data['UserID'].values:
            user_data.loc[user_data['UserID'] == user_id, 'Choc'] = answer
        else:
            new_entry = pd.DataFrame({'UserID': [user_id], 'Choc': [answer]})
            user_data = pd.concat([user_data, new_entry], ignore_index=True)

        await message.answer('Вы пробовали когда-либо кофеин?', reply_markup=yes_no_keyboard)
        await state.set_state(TestStates.Q4)

    @router.message(TestStates.Q4, lambda message: message.text.lower() in ['да', 'нет'])
    async def handle_caff(message: types.Message, state: FSMContext):
        user_id = message.from_user.id
        answer = 1 if message.text.lower() == 'да' else 0

        global user_data
        if user_id in user_data['UserID'].values:
            user_data.loc[user_data['UserID'] == user_id, 'Caff'] = answer
        else:
            new_entry = pd.DataFrame({'UserID': [user_id], 'Caff': [answer]})
            user_data = pd.concat([user_data, new_entry], ignore_index=True)

        await normalize_data()

        ans = await predict(user_id)

        s = ''
        for test, answer in ans.items():
            s += f"вероятность употребления {test} составляет {answer[0][1]:.2f}\n"

        await message.answer(f"Спасибо за ваши ответы!\n {s}", reply_markup=ReplyKeyboardRemove())

    dp.include_router(router)

    await dp.start_polling(bot)


if __name__ == "__main__":
    unpack_models()

    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print('Bot stopped!')
