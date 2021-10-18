from pathlib import Path

from app import main


def read_token() -> str:
    token_path = Path(__file__).parent / 'token'
    with token_path.open() as token_file:
        return token_file.readline()


if __name__ == '__main__':
    TOKEN = read_token()
    main(TOKEN)
