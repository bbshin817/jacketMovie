import xml.etree.ElementTree as ET

class Lyrics:
	lyrics = []

	def __init__(self):
		self.lyrics = []

	def addLyric(self, words, begin, end, syllables, isSmall = False):
		self.lyrics.append({
			'startTimeMs' : parseDuration(begin),
			'words' : words,
			'syllables' : syllables,
			'endTimeMs' : parseDuration(end),
			'isSmall' : isSmall
		})
		return

class Syllables:
	words = ''
	syllables = []

	def __init__(self):
		self.words = ''
		self.syllables = []

	def addSyllable(self, word, begin, end, isSpace):
		self.words += word
		self.syllables.append({
			'startTimeMs' : parseDuration(begin),
			'numChars' : len(word),
			'endTimeMs' : parseDuration(end) - (1 if isSpace else 0)
		})
		if (isSpace != 'None'):
			self.words += ' '
			self.syllables.append({
				'startTimeMs' : parseDuration(end) - (1 if isSpace else 0),
				'numChars' : 1,
				'endTimeMs' : parseDuration(end)
			})

def parseDuration(duration: str) -> int:
	duration = duration.replace('s', '')
	parts = duration.split(":")
	if len(parts) == 3:
		minutes = int(parts[1])
		seconds = float(parts[2])
	elif len(parts) == 2:
		minutes = int(parts[0])
		seconds = float(parts[1])
	else:
		minutes = 0
		seconds = float(duration)
	total_ms = int((minutes * 60 + seconds) * 1000)
	return total_ms

def parseSpotify(ttml):
	xml = ET.fromstring(ttml)
	divIndex = 0
	lyrics = Lyrics()
	for div in xml[1]:
		for p in div:
			if (p.text):
				lyrics.addLyric(p.text, p.get('begin'), p.get('end'), [])
			else:
				begin = p.get('begin')
				isSmall = False
				syllables = Syllables()
				for span in p:
					if (span.text):
						syllables.addSyllable(span.text, span.get('begin'), span.get('end'), repr(span.tail))
						end = span.get('end')
					else:
						isSmall = True
						lyrics.addLyric(syllables.words, begin, end, syllables.syllables)
						begin = span[0].get('begin')
						syllables = Syllables()
						for childSpan in span:
							syllables.addSyllable(childSpan.text, childSpan.get('begin'), childSpan.get('end'), repr(childSpan.tail))
						end = span[len(span) - 1].get('end')
				lyrics.addLyric(syllables.words, begin, end, syllables.syllables, isSmall)
		if (divIndex < len(xml[1]) - 1):
			lyrics.addLyric('', div.get('end'), xml[1][divIndex + 1].get('begin'), [])
		divIndex += 1
	return lyrics.lyrics

def toSpotifyJson(dict):
	divs = dict['tt']['body']['div']
	lyrics = []
	divIndex = 0
	for div in divs:
		if isinstance(div['p'], list):
			for p in div['p']:
				words = ''
				syllables = []
				for span in p['span']:
					words += span['#text']
					syllables.append({
						'startTimeMs' : parseDuration(span['@begin']),
						'numChars' : len(span['#text']),
						'endTimeMs' : parseDuration(span['@end'])
					})
				lyrics.append({
					'startTimeMs' : parseDuration(p['span'][0]['@begin']),
					'words' : words,
					'syllables' : syllables,
					'endTimeMs' : parseDuration(p['span'][len(p['span']) - 1]['@end'])
				})
		else:
			words = ''
			syllables = []
			for span in div['p']['span']:
				words += span['#text']
				syllables.append({
					'startTimeMs' : parseDuration(span['@begin']),
					'numChars' : len(span['#text']),
					'endTimeMs' : parseDuration(span['@end'])
				})
			lyrics.append({
				'startTimeMs' : parseDuration(div['p']['span'][0]['@begin']),
				'words' : words,
				'syllables' : syllables,
				'endTimeMs' : parseDuration(div['p']['span'][len(div['p']['span']) - 1]['@end'])
			})
		if (divIndex < len(divs) - 1):
			lyrics.append({
				'startTimeMs' : parseDuration(div['@end']),
				'words' : '',
				'syllables' : [],
				'endTimeMs' : parseDuration(divs[divIndex + 1]['@begin'])
			})
		else:
			lyrics.append({
				'startTimeMs' : parseDuration(div['@end']),
				'words' : '',
				'syllables' : [],
				'endTimeMs' : parseDuration(dict['tt']['body']['@dur'])
			})
		divIndex += 1
	return {
		'lyric' : lyrics
	}

if __name__ == '__main__':
	with open('raw-lyric.xml', 'r', encoding='utf-8') as f:
		ttml = f.read()
		f.close()
	parseSpotify(ttml)