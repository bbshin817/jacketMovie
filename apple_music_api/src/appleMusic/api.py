import requests
from . import authorization, config

class AppleMusicAPI:
	session = None

	def __init__(self):
		self.session = requests.Session()
		self.session.headers = config.REQUEST_HEADER
		self.session.headers.update({
			'Authorization' : f'Bearer {authorization.getAccessToken()}',
			'Media-User-Token' : authorization.getMediaUserToken()
		})
		self.getSubscriptionState()

	def getSubscriptionState(self):
		response = self.session.get(
			'https://amp-api.music.apple.com/v1/me/account?meta=subscription'
		)
		assert response, 'サブスクリプション状態の取得に失敗しました'
		isSubscriptionActive = response.json()['meta']['subscription']['active']
		if not (isSubscriptionActive):
			raise Exception('有効なApple Music サブスクリプションが存在するmediaUserTokenを指定してください')
		return isSubscriptionActive
	
	def getSyllableLyricAsTTML(self, trackId):
		response = self.session.get(
			f'https://amp-api.music.apple.com/v1/catalog/jp/songs/{trackId}/syllable-lyrics'
		)
		assert response, '歌詞の取得に失敗しました'
		response = response.json()
		return response['data'][0]['attributes']['ttml']
	
	def getCatalog(self, trackId):
		response = self.session.get(
			f'https://amp-api.music.apple.com/v1/catalog/jp/songs/{trackId}?l=ja&include=albums&format%5Bresources%5D=map&platform=web'
		)
		assert response, '歌詞の取得に失敗しました'
		response = response.json()
		return response

if __name__ == '__main__':
	API = AppleMusicAPI()
	API.getSubscriptionState()
	# print(API.getSyllableLyricAsXml('1440788515'))