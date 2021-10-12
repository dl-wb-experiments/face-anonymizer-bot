from telegram.ext import (CallbackQueryHandler, ConversationHandler, CommandHandler, Filters,
                          MessageHandler, Updater)

from app.constants import BotConversationState
from app.handlers import (select_anonymization_type_handler, stop_handler, handle_anonymization_type_handler,
                          start_handler, process_media_handler)


def main() -> None:
    """Start the bot."""
    with open('app/token') as token_file:
        token = token_file.readline()

    updater = Updater(token)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    conversation_handler = ConversationHandler(
        entry_points=[CommandHandler('start', select_anonymization_type_handler)],

        states={
            BotConversationState.select_anonymization_type: [
                CallbackQueryHandler(handle_anonymization_type_handler),
            ],
            BotConversationState.upload_media: [
                MessageHandler(Filters.photo | Filters.video | Filters.video_note, process_media_handler)
            ],
        },

        fallbacks=[CommandHandler('stop', stop_handler)],
    )
    updater.dispatcher.add_handler(conversation_handler)

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler('start', start_handler))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()
