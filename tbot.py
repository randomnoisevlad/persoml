import time
import telebot
import requests

TOKEN = ""
KERAS_REST_API_URL = ""

bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 'Hey Start')


@bot.message_handler(commands=['info'])
def send_welcome(message):
    info = ('Hey Info')
    bot.reply_to(message, info)


import uuid

@bot.message_handler(content_types=["photo"])
def answer_photo(message):
    photo = bot.get_file(message.photo[-1].file_id)
    # URL direction to image
    photo_url = "https://api.telegram.org/file/bot{0}/{1}".format(
        TOKEN, photo.file_path)
    # Computer Vision parameters
    r = requests.get(photo_url)
    file_name = str(uuid.uuid4()) + '.png'
    if r.status_code == 200:
        with open('temp/' + file_name, 'wb') as f:
            f.write(r.content)
    else:
        bot.reply_to(message, 'something fails...')
        return

    img = open('temp/' + file_name, 'rb')

    #img = open('inpred.png', 'rb')

    payload = {"image":img}

    bot.send_chat_action(message.chat.id, 'typing')
    try:
        r = requests.post(KERAS_REST_API_URL, files=payload).json()
    except:
        bot.reply_to(message, 'something fails....')
    print(r)
    time.sleep(1)

    img_path = None
    try:
        if r['success']:
            img_path = r['result_path']
            img_result = open(img_path, 'rb')
            bot.reply_to(message, photo_url)
            bot.send_photo(message.chat.id, img_result, reply_to_message_id=message.message_id)

            img_path = r['mask_path']
            img_result = open(img_path, 'rb')
            bot.reply_to(message, photo_url)
            bot.send_photo(message.chat.id, img_result, reply_to_message_id=message.message_id)
      
            img_path = r['cg_path']
            img_result = open(img_path, 'rb')
            bot.reply_to(message, photo_url)
            bot.send_photo(message.chat.id, img_result, reply_to_message_id=message.message_id)
        else:
            bot.reply_to(message, 'something fails...')
    except:
        bot.reply_to(message, 'something fails...')

@bot.message_handler(func=lambda m: True)
def reply_all(message):
    if message.chat.type == "private":
        bot.reply_to(message, 'Please send me an image so I can describe it!')


bot.polling(none_stop=True)


while True:
    time.sleep(5)


