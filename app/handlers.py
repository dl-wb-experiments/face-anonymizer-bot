from pathlib import Path
from tempfile import TemporaryDirectory

from telegram import Update, ReplyKeyboardRemove, File, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, ConversationHandler

from app.constants import BotConversationState, AnonymizeType
from app.face_anonymizer import BlurFaceAnonymizer, EmotionFaceAnonymizer


def start_handler(update: Update, _) -> int:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
    )
    return BotConversationState.select_anonymization_type


def select_anonymization_type_handler(update: Update, _) -> int:
    """Sends a message with two inline buttons attached."""
    keyboard = [
        [
            InlineKeyboardButton('Blur', callback_data=AnonymizeType.blur.value),
            InlineKeyboardButton('Emoji', callback_data=AnonymizeType.emoji.value),
        ],
    ]

    anonymization_reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Choose anonymization type:',
                              reply_markup=anonymization_reply_markup)

    return BotConversationState.select_anonymization_type


def handle_anonymization_type_handler(update: Update, context: CallbackContext) -> BotConversationState:
    query = update.callback_query
    query.answer()
    selected_anonymization_type = AnonymizeType(query.data)
    context.user_data['anonymization_type'] = selected_anonymization_type
    query.edit_message_text(text=f'Selected anonymization type: {selected_anonymization_type}')
    query.message.reply_text('Upload photo or video you want to anonymize')
    return BotConversationState.upload_media


def process_media_handler(update: Update, context: CallbackContext) -> int:
    raw_anonymization_type = context.user_data.get('anonymization_type')
    try:
        selected_anonymization_type = AnonymizeType(raw_anonymization_type)
    except ValueError:
        update.message.reply_text('Anonymization type is not selected')
        return BotConversationState.select_anonymization_type

    if selected_anonymization_type == AnonymizeType.blur:
        FaceAnonymizer = BlurFaceAnonymizer
    elif selected_anonymization_type == AnonymizeType.emoji:
        FaceAnonymizer = EmotionFaceAnonymizer
    else:
        update.message.reply_text(f'Anonymization type is not supported: {selected_anonymization_type.value}')
        return BotConversationState.select_anonymization_type

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
        update.message.reply_text('Bad message media type - only photos and videos are supported')
        return BotConversationState.select_anonymization_type

    update.message.reply_text('Processing...')

    file: File = context.bot.getFile(file_id)
    with TemporaryDirectory() as tmpdir:
        file_name = Path(file.file_path).name
        file_extension = Path(file.file_path).suffix
        input_file_path = str(Path(tmpdir) / file_name)
        file.download(custom_path=input_file_path)

        output_file_name = f'{file_id}_anonymized{file_extension}'
        output_file_path = str(Path(tmpdir) / output_file_name)

        face_anonymizer_handler(input_file_path=input_file_path, output_file_path=output_file_path)
        with open(output_file_path, mode='rb') as output_file:
            update.message.reply_text('Here is your anonymized media:')
            send_file_handler(update.effective_chat.id, output_file)
    return select_anonymization_type_handler(update, context)


def stop_handler(update: Update, _) -> int:
    """Cancels and ends the conversation."""
    update.message.reply_text(
        'Bye! I hope we can talk again some day.', reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END
