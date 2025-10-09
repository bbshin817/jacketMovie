import re

def getTrackIdFromUrl(url: str):
	url = re.sub(r'https?:\/\/', '', url)
	url = url.split('/')
	if (url[2] == 'song'):
		trackId = re.search(r'\d+', url[4])
		if (trackId):
			return trackId.group()
		else:
			return None
	return None