import requests, re
from pathlib import Path
from . import config

def getAccessToken():
	response = requests.get(
		'https://music.apple.com',
		headers=config.REQUEST_HEADER
	)
	assert response.status_code == 200, 'htmlの取得に失敗'
	match = re.search(r'assets/index.[0-9a-z]+.js', response.text)
	assert match, 'module javascriptの検出に失敗'
	response = requests.get(
		f'https://music.apple.com/{match.group()}',
		headers=config.REQUEST_HEADER
	)
	assert response.status_code == 200, 'module javascriptの取得に失敗'
	match = re.search(r'ey[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', response.text)
	assert match, 'access Tokenの抽出に失敗'
	return match.group()

def getMediaUserToken():
	base_dir = Path(__file__).resolve().parent
	with open(f'{base_dir}/../../media-user-token', 'r', encoding='utf-8') as f:
		mediaUserToken = f.read()
		f.close()
	return mediaUserToken

if __name__ == '__main__':
	print(getAccessToken(), getMediaUserToken())