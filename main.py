#!/usr/bin/env python

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from telegram import Update, ReplyKeyboardRemove, File, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler, \
    CallbackQueryHandler

from app.constants import BotConversationState, AnonymizeType
from app.face_anonymizer import BlurFaceAnonymizer, EmotionFaceAnonymizer

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


def start(update: Update, context: CallbackContext) -> int:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
    )
    return BotConversationState.select_anonymization_type.value


def select_anonymization_type(update: Update, context: CallbackContext) -> int:
    """Sends a message with three inline buttons attached."""
    keyboard = [
        [
            InlineKeyboardButton('Blur', callback_data=AnonymizeType.blur.value),
            InlineKeyboardButton('Emoji', callback_data=AnonymizeType.emoji.value),
        ],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Choose anonymization type:', reply_markup=reply_markup)
    return BotConversationState.select_anonymization_type.value


def handle_anonymization_type(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    query.answer()
    selected_anonymization_type = query.data
    context.user_data['anonymization_type'] = selected_anonymization_type
    query.edit_message_text(text=f'Selected anonymization type: {selected_anonymization_type}')
    query.message.reply_text('Upload photo or video you want to anonymize')
    return BotConversationState.upload_media.value


def process_media(update: Update, context: CallbackContext) -> int:
    selected_anonymization_type = context.user_data['anonymization_type']
    if not selected_anonymization_type:
        update.message.reply_text('Anonymization type is not selected')
        return BotConversationState.select_anonymization_type.value

    FaceAnonymizer = BlurFaceAnonymizer if selected_anonymization_type == AnonymizeType.blur.value \
        else EmotionFaceAnonymizer

    message = update.message
    message_photo = message.photo
    message_video = message.video or message.video_note
    if message_photo:
        file_id = message_photo[-1].file_id
        face_anonymizer_handler = FaceAnonymizer.process_image
        send_file_handler = context.bot.send_photo
    elif message_video:
        file_id = message_video.file_id
        face_anonymizer_handler = FaceAnonymizer.process_video
        send_file_handler = context.bot.send_video
    else:
        raise Exception('Bad message media type - only photos and videos are supported')

    update.message.reply_text('Processing...')

    file: File = context.bot.getFile(file_id)
    with TemporaryDirectory() as tmpdir:
        file_name = Path(file.file_path).name
        file_extension = Path(file.file_path).suffix
        input_file_path = str(Path(tmpdir) / file_name)
        file.download(custom_path=input_file_path)

        output_file_name = f'{file_id}_anonymized{file_extension}'
        output_file_path = str(Path(tmpdir) / output_file_name)

        face_anonymizer_handler(input_file_path=input_file_path,
                                output_file_path=output_file_path)
        with open(output_file_path, mode='rb') as output_file:
            update.message.reply_text('Here is your anonymized media:')
            send_file_handler(update.effective_chat.id, output_file)


def cancel(update: Update, context: CallbackContext) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text(
        'Bye! I hope we can talk again some day.', reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    token = os.environ.get('TELEGRAM_TOKEN')
    updater = Updater(token)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    conversation_handler = ConversationHandler(
        entry_points=[CommandHandler('start', select_anonymization_type)],

        states={
            # BotConversationState.start.value: [
            #     CallbackQueryHandler(handle_anonymization_type),
            # ],
            BotConversationState.select_anonymization_type.value: [
                CallbackQueryHandler(handle_anonymization_type),
            ],
            BotConversationState.upload_media.value: [
                MessageHandler(Filters.photo | Filters.video | Filters.video_note, process_media)
            ],
        },

        fallbacks=[CommandHandler('cancel', cancel)],
    )
    updater.dispatcher.add_handler(conversation_handler)

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler('start', start))
    # TODO Add help command handler
    # dispatcher.add_handler(CommandHandler('help', help_command))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
