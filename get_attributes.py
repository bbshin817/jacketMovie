import sys, json, requests
from apple_music_api.src.appleMusic.api import AppleMusicAPI
from apple_music_api.src.print import error
from apple_music_api.src.parse import getTrackIdFromUrl
from apple_music_api.src.formatter import parseSpotify

if __name__ == '__main__':
	if (len(sys.argv) == 1):
		error('有効なトラックURLを渡してください')
	API = AppleMusicAPI()
	try:
		trackId = getTrackIdFromUrl(sys.argv[1])
		if trackId is None:
			error('有効なトラックURLを渡してください')
		rawLyrics = API.getSyllableLyricAsTTML(trackId)
		with open('./lyric.json', 'w', encoding='utf-8') as f:
			lyrics = parseSpotify(rawLyrics)
			json.dump(lyrics, f, ensure_ascii=False, indent=4)
			f.close()
		catalog = API.getCatalog(trackId)
		artworkQuery = {
			'{w}' : 600,
			'{h}' : 600,
			'{f}' : 'jpg'
		}
		trackAttributes = catalog['resources']['songs'][trackId]['attributes']
		artworkUrl = trackAttributes['artwork']['url']
		for old, new in artworkQuery.items():
			artworkUrl = artworkUrl.replace(old, f'{new}')
		with open('./cover.jpg', 'wb') as f:
			f.write(requests.get(artworkUrl).content)
			f.close()
		with open('./attributes.json', 'w', encoding='utf-8') as f:
			json.dump({
				'trackName' : trackAttributes['name'],
				'artistName' : trackAttributes['artistName']
			}, f, ensure_ascii=False, indent=4)
			f.close()
	except Exception as e:
		error(str(e))